"""
认知评估与执行管道 (Cognition Pipeline) v1.1
=====================================

用户输入洞见 → AI分析评估 → 判断是否执行 → 改代码 → 记录

工作流:
  1. 用户输入洞见文本
  2. AI评估: 价值/工作量/风险/回退方案
  3. 输出评估报告
  4. 如果决策=执行 -> AI直接改代码
  5. 如果决策=暂缓 -> 写入认知待办
  6. 所有操作记录到 project_cognition.md

使用方式:
  python cognition_pipeline.py run "你的洞见"       评估+执行
  python cognition_pipeline.py eval "你的洞见"      仅评估不执行
  python cognition_pipeline.py optimize             自迭代模式(从待办取任务)
  python cognition_pipeline.py status               查看所有认知状态

回退机制:
  每次代码修改前自动备份 (backups/)
  python cognition_pipeline.py rollback <cid/backup_id>
"""

import sys, os, json, re, datetime, shutil, subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
COGNITION_DB = PROJECT_ROOT / 'project_cognition.md'
BACKUP_DIR = PROJECT_ROOT / 'backups'
TODO_FILE = PROJECT_ROOT / 'cognition_todo.md'
BACKUP_DIR.mkdir(exist_ok=True)

EVAL_TEMPLATE = {
    "cognition_id": "auto-{timestamp}",
    "cognition_text": "",
    "analysis": {
        "summary": "一句话总结洞见核心",
        "relevance_to_codebase": "与哪些模块相关",
        "technical_feasibility": "是否技术上可做",
        "data_requirements": "需要什么数据/传感器",
        "current_limitations": "当前有什么限制"
    },
    "evaluation": {
        "value_score": 0,
        "effort_hours": 0.0,
        "risk_level": "low",
        "risk_factors": [],
        "rollback_plan": "如何回退"
    },
    "decision": "defer",
    "execution_plan": {
        "files_to_modify": [],
        "changes_summary": "",
        "testing_method": ""
    },
    "rollback": {
        "backup_files": [],
        "restore_command": ""
    }
}

# ============================================
# 文件操作
# ============================================

def backup_files(file_paths):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_id = f"backup_{ts}"
    backup_path = BACKUP_DIR / backup_id
    backup_path.mkdir(exist_ok=True)
    for fp in file_paths:
        fpath = os.path.join(PROJECT_ROOT, fp) if not os.path.isabs(fp) else fp
        if os.path.exists(fpath):
            shutil.copy2(fpath, backup_path / os.path.basename(fp))
    return backup_id, str(backup_path)

def restore_backup(backup_id):
    backup_path = BACKUP_DIR / backup_id
    if not backup_path.exists():
        return False, f"备份 {backup_id} 不存在"
    for f in backup_path.iterdir():
        dest = PROJECT_ROOT / f.name
        shutil.copy2(str(f), str(dest))
    return True, f"已从 {backup_id} 恢复"

def record_cognition(cid, text, value_score, effort_hours, risk_level,
                      decision, files_changed, success=True):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    status = '[OK]' if success else '[FAIL]'
    entry = f"""
### {cid} ({ts})
- **洞见**: {text}
- **价值**: {value_score}/10 | **工时**: {effort_hours}h | **风险**: {risk_level}
- **决策**: {decision} | **状态**: {status}
- **文件**: {', '.join(files_changed)}
- **回退**: `python cognition_pipeline.py rollback {cid}`
"""
    with open(COGNITION_DB, 'a', encoding='utf-8') as f:
        f.write(entry)

# ============================================
# 扫描待办
# ============================================

def scan_todo():
    if not TODO_FILE.exists():
        return []
    with open(TODO_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    todos = []
    current = {}
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('## '):
            if current:
                todos.append(current)
            current = {'timestamp': line.replace('## ', '').strip()}
        elif line.startswith('- **洞见**'):
            current['insight'] = line.replace('- **洞见**: ', '').strip()
        elif line.startswith('- **待做**'):
            current['action'] = line.replace('- **待做**: ', '').strip()
    if current:
        todos.append(current)
    return todos

def remove_todo(insight_text):
    """从待办移除已执行项"""
    if not TODO_FILE.exists():
        return
    with open(TODO_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    # 找到并注释（不删除，保留历史）
    new_lines = []
    skip = False
    for line in lines:
        if insight_text in line:
            skip = True
            new_lines.append('<!-- ' + line.rstrip() + ' (DONE) -->\n')
        elif skip and line.strip().startswith('-'):
            new_lines.append('<!-- ' + line.rstrip() + ' -->\n')
        else:
            skip = False
            new_lines.append(line)
    with open(TODO_FILE, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

# ============================================
# CLI 命令
# ============================================

def cmd_status():
    print(f"\n认知数据库: {COGNITION_DB}")
    print(f"备份目录: {BACKUP_DIR}/")
    print(f"待办文件: {TODO_FILE}")
    with open(COGNITION_DB, 'r', encoding='utf-8') as f:
        content = f.read()
    executed = content.count('[OK]')
    failed = content.count('[FAIL]')
    print(f"\n统计: 已执行={executed} 失败={failed}")
    backups = sorted(BACKUP_DIR.iterdir()) if BACKUP_DIR.exists() else []
    if backups:
        print(f"最近备份: {backups[-1].name}")

def cmd_rollback(cid):
    backup_id = None
    for b in sorted(BACKUP_DIR.iterdir()):
        if cid in b.name:
            backup_id = b.name
            break
    if backup_id:
        ok, msg = restore_backup(backup_id)
        print(f"[{'OK' if ok else 'FAIL'}] {msg}")
    else:
        print(f"未找到 {cid} 的备份，尝试git...")
        subprocess.run(['git', 'checkout', '--', '.'], cwd=PROJECT_ROOT)
        print("git checkout 恢复完成")

def cmd_optimize():
    todos = scan_todo()
    if not todos:
        print("待办列表为空")
        return []
    print(f"发现 {len(todos)} 个待办:")
    for i, t in enumerate(todos, 1):
        print(f"  [{i}] {t.get('timestamp','')} | {t.get('insight','')[:60]}")
    selected = os.environ.get('COGNITION_SELECTED', '')
    if selected:
        idx = int(selected) - 1
        if 0 <= idx < len(todos):
            t = todos[idx]
            print(f"\n选择 [{selected}]: {t.get('insight','')[:60]}")
            print("--- AI执行阶段 ---")
            print("1. 备份涉及文件")
            print("2. 修改代码")
            print("3. 验证")
            print("4. 更新认知数据库")
            print("5. 从待办移除")
            return [t]
        else:
            print(f"索引超出范围(1-{len(todos)})")
    else:
        print("\nAI: 请评估以上待办，选择一项执行。")
        print("   设置环境变量 COGNITION_SELECTED=<序号> 选择任务")
    return []

def cmd_eval(text):
    print(f"\n{'='*60}")
    print(f"评估: {text[:80]}")
    print(f"{'='*60}")
    print("(AI在此输出评估报告)")

def cmd_run(text):
    cmd_eval(text)
    print("\n--- AI执行阶段 ---")
    print("(AI在此执行修改)")

# ============================================
# 入口
# ============================================

if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if action == 'status':
        cmd_status()
    elif action == 'rollback':
        cmd_rollback(sys.argv[2] if len(sys.argv) > 2 else 'last')
    elif action == 'eval':
        text = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read().strip()
        cmd_eval(text)
    elif action == 'run':
        text = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read().strip()
        cmd_run(text)
    elif action == 'optimize':
        cmd_optimize()
    else:
        print(__doc__)
