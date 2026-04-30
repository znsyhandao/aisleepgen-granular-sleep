"""
认知评估与执行管道 v2 - 自迭代核心
====================================

用法:
  python cognition_pipeline.py status                查看状态
  python cognition_pipeline.py rollback <cid>        回退
  python cognition_pipeline.py optimize              自迭代(扫描待办+评估+选任务)
  python cognition_pipeline.py evolve "目标"          思维轨迹进化(多方案并行+残差重组)
  python cognition_pipeline.py run "洞见"            评估+执行
  python cognition_pipeline.py eval "洞见"           仅评估

自迭代流程:
  1. AI扫描 cogniton_todo.md 待办列表
  2. AI评估每项的 价值/工时/风险/可行性
  3. AI选择最高价值/最可行的任务
  4. 备份涉及文件 -> 执行修改 -> 验证 -> 记录 -> 从待办移除
  
  如果修改失败或验证不通过: 自动回退备份

回退: python cognition_pipeline.py rollback <认知ID>
"""

import sys, os, json, re, datetime, shutil, subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
COGNITION_DB = PROJECT_ROOT / 'project_cognition.md'
BACKUP_DIR = PROJECT_ROOT / 'backups'
TODO_FILE = PROJECT_ROOT / 'cognition_todo.md'
BACKUP_DIR.mkdir(exist_ok=True)

# ============================================
# 备份/回退
# ============================================

def backup_files(file_paths):
    ts = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
    backup_id = f"bkp_{ts}"
    bp = BACKUP_DIR / backup_id; bp.mkdir(exist_ok=True)
    for fp in file_paths:
        fpath = Path(fp) if os.path.isabs(fp) else PROJECT_ROOT / fp
        if fpath.exists():
            shutil.copy2(str(fpath), str(bp / fpath.name))
    return backup_id

def restore_backup(backup_id):
    bp = BACKUP_DIR / backup_id
    if not bp.exists():
        return False, f"backup {backup_id} not found"
    for f in bp.iterdir():
        dest = PROJECT_ROOT / f.name
        shutil.copy2(str(f), str(dest))
    return True, f"restored from {backup_id}"

def git_snapshot(msg=""):
    try:
        subprocess.run(['git', 'add', '-A'], cwd=PROJECT_ROOT,
                       capture_output=True, timeout=10)
        r = subprocess.run(['git', 'commit', '--allow-empty', '-m',
                            msg or f'bkp {datetime.datetime.now().strftime("%H%M%S")}'],
                           cwd=PROJECT_ROOT, capture_output=True, timeout=10, text=True)
        return r.stdout.strip()[:60]
    except:
        return 'git snap failed'

# ============================================
# 记录系统
# ============================================

def record_cognition(cid, text, vs=0, eh=0, rl='low', decision='defer',
                      files=None, ok=True, note=''):
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    st = '[OK]' if ok else '[FAIL]'
    files = files or []
    entry = f"""
| {ts} | {cid} | {st} | v{vs} w{eh}h r{rl} | {text[:50]} | {' '.join(files)} |"""
    with open(COGNITION_DB, 'a', encoding='utf-8') as f:
        f.write(entry)

def init_db():
    if not COGNITION_DB.exists():
        with open(COGNITION_DB, 'w', encoding='utf-8') as f:
            f.write("""# 认知数据库

## 锚点索引
| 锚点 | 位置 | 状态 |
|------|------|------|
| SWA | granular/sleep_metrics.py | active |
| narrative | granular/sleep_metrics.py | active |

## 注入历史
| 时间 | ID | 状态 | 评估 | 洞见 | 文件 |
|------|----|------|------|------|------|
""")

# ============================================
# 待办系统
# ============================================

def scan_todo():
    if not TODO_FILE.exists():
        return []
    with open(TODO_FILE, 'r', encoding='utf-8') as f:
        content = f.read()
    todos = []
    cur = {}
    for line in content.split('\n'):
        ls = line.strip()
        if ls.startswith('## '):
            if cur: todos.append(cur)
            cur = {'timestamp': ls.replace('## ', '')}
        elif ls.startswith('- **洞见**'):
            cur['insight'] = ls.replace('- **洞见**: ', '')
        elif ls.startswith('- **待做**'):
            cur['action'] = ls.replace('- **待做**: ', '')
    if cur: todos.append(cur)
    return todos

def mark_todo_done(insight_substr):
    if not TODO_FILE.exists():
        return
    with open(TODO_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    new_lines = []
    marking = False
    for line in lines:
        if insight_substr in line and line.strip().startswith('-'):
            new_lines.append('<!-- ' + line.rstrip() + ' (DONE) -->\n')
            marking = True
        elif marking and line.strip().startswith('-'):
            new_lines.append('<!-- ' + line.rstrip() + ' -->\n')
        else:
            marking = False
            new_lines.append(line)
    with open(TODO_FILE, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

# ============================================
# CLI
# ============================================

def cmd_status():
    init_db()
    todos = scan_todo()
    with open(COGNITION_DB, 'r', encoding='utf-8') as f:
        db = f.read()
    executed = db.count('[OK]')
    failed = db.count('[FAIL]')
    backups = sorted(BACKUP_DIR.iterdir()) if BACKUP_DIR.exists() else []
    print(f"\n{'='*50}")
    print(f"认知管道 v2")
    print(f"{'='*50}")
    print(f"已执行: {executed}  失败: {failed}  待办: {len(todos)}")
    print(f"备份数: {len(backups)}")
    if todos:
        print(f"\n待办:")
        for i, t in enumerate(todos, 1):
            print(f"  [{i}] {t.get('timestamp','')}: {t.get('insight','?')[:60]}")

def cmd_rollback(cid):
    for b in sorted(BACKUP_DIR.iterdir()):
        if cid in b.name:
            ok, msg = restore_backup(b.name)
            print(f"[{'OK' if ok else 'FAIL'}] {msg}")
            return
    print(f"未找到 {cid} 备份, 尝试 git checkout...")
    subprocess.run(['git', 'checkout', '--', '.'], cwd=PROJECT_ROOT)

def cmd_optimize():
    """自迭代入口 - AI负责评估+选择+执行"""
    init_db()
    todos = scan_todo()
    if not todos:
        print("待办为空")
        return

    print(f"\n{'='*50}")
    print(f"自迭代模式 - {len(todos)} 个待办")
    print(f"{'='*50}\n")

    for i, t in enumerate(todos, 1):
        ins = t.get('insight', '?')
        act = t.get('action', '')
        print(f"  [{i}] {ins}")
        print(f"       待做: {act}")
        print()

    print(f"{'='*50}")
    print("AI 评估阶段:")
    print("  评估每项: 价值(1-10) / 工时(h) / 风险 / 可行性")
    print("  选择一项: 综合分析后选择")
    print()
    print("AI 执行阶段:")
    print("  1. 备份涉及文件")
    print("  2. 修改代码")
    print("  3. 验证(语法+逻辑)")
    print("  4. 回退失败 or 记录并完成")
    print(f"{'='*50}")

def cmd_evolve(goal_text):
    """
    L3: 思维轨迹进化
    从同一个目标出发，生成多个竞争方案 → 验证 → 选优 → 残差重组
    """
    init_db()

    print(f"\n{'='*60}")
    print(f"L3 思维轨迹进化")
    print(f"{'='*60}")
    print(f"目标: {goal_text[:80]}")
    print()

    # Phase 1: 生成3个竞争方案
    print("--- Phase 1: 并行方案生成 ---")
    print("方案A: 最小改动（当前代码基上直接修改）")
    print("方案B: 全新实现（重写关键模块）")
    print("方案C: 混合策略（复用+替换）")
    print()

    # Phase 2: 各自验证
    print("--- Phase 2: 并行验证 ---")
    print("方案A: 语法检查 → 单元测试 → 覆盖率")
    print("方案B: 语法检查 → 单元测试 → 覆盖率")
    print("方案C: 语法检查 → 单元测试 → 覆盖率")
    print()

    # Phase 3: 选优
    print("--- Phase 3: 选择最优 ---")
    print("对比指标: 性能/可读性/可扩展性/风险")
    print("选择: [AI在此输出]")
    print()

    # Phase 4: 残差重组
    print("--- Phase 4: 残差重组 ---")
    print("落选方案中有价值的思路合并到最优方案")
    print("合并后: [AI在此输出]")
    print()

    # Phase 5: 最终输出
    print("--- Phase 5: 最终输出 ---")
    print("方案来源: A + B的[C特征] + C的[D特征]")
    print("文件: [AI在此填充]")
    print()

    return ['A', 'B', 'C']  # AI在方法内实际执行

# ============================================
# 入口
# ============================================

if __name__ == '__main__':
    ac = sys.argv[1] if len(sys.argv) > 1 else 'help'

    if ac == 'status':
        cmd_status()
    elif ac == 'rollback':
        cmd_rollback(sys.argv[2] if len(sys.argv) > 2 else 'last')
    elif ac == 'optimize':
        cmd_optimize()
    elif ac == 'evolve':
        text = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read().strip()
        if not text:
            print("需要目标描述: python cognition_pipeline.py evolve \"你的目标\"")
            sys.exit(1)
        cmd_evolve(text)
    elif ac == 'eval':
        cmd_run_eval(sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read().strip(), False)
    elif ac == 'run':
        cmd_run_eval(sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read().strip(), True)
    else:
        print(__doc__)
