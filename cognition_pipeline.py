"""
认知评估与执行管道 (Cognition Pipeline)
=====================================

用户输入洞见 → AI分析评估 → 判断是否执行 → 改代码 → 记录

工作流:
  1. 用户输入洞见文本
  2. AI评估: 价值/工作量/风险/回退方案
  3. 输出评估报告
  4. 如果决策=执行 → AI直接改代码
  5. 如果决策=暂缓 → 写入认知待办
  6. 所有操作记录到 project_cognition.md

使用方式:
  python cognition_pipeline.py run "你的洞见"
  python cognition_pipeline.py eval "你的洞见"  # 仅评估不执行
  python cognition_pipeline.py status           # 查看所有认知状态
  
回退机制:
  每次代码修改前自动备份 (backups/)
  如果修改有问题: python cognition_pipeline.py rollback <id>
"""

import sys, os, json, re, datetime, shutil, subprocess
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.absolute()
COGNITION_DB = PROJECT_ROOT / 'project_cognition.md'
BACKUP_DIR = PROJECT_ROOT / 'backups'
TODO_FILE = PROJECT_ROOT / 'cognition_todo.md'
BACKUP_DIR.mkdir(exist_ok=True)

# 认知注入记录
class CognitionRecord:
    def __init__(self, cid, text, value_score, effort_hours, risk_level, 
                 decision, files_changed, rollback_cmd, timestamp):
        self.cid = cid
        self.text = text
        self.value_score = value_score      # 1-10
        self.effort_hours = effort_hours    # 预计工时
        self.risk_level = risk_level        # low/medium/high
        self.decision = decision            # execute / defer / reject
        self.files_changed = files_changed  # list of file paths
        self.rollback_cmd = rollback_cmd   # rollback command
        self.timestamp = timestamp
        
    def to_dict(self):
        return self.__dict__

# ============================================
# 1. 评估引擎 - 由AI调用来分析洞见
# ============================================

# 评估模板 - AI填充此结构
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
        "value_score": 0,          # 1-10 对产品的核心竞争力提升
        "effort_hours": 0.0,       # 预计纯开发工时
        "risk_level": "low",       # low / medium / high
        "risk_factors": ["风险描述"],
        "rollback_plan": "如何回退"
    },
    "decision": "defer",           # execute / defer / reject
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
# 2. 文件操作 - 备份/恢复/修改
# ============================================

def backup_files(file_paths):
    """修改前备份文件"""
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
    """从备份恢复文件"""
    backup_path = BACKUP_DIR / backup_id
    if not backup_path.exists():
        return False, f"备份 {backup_id} 不存在"
    
    for f in backup_path.iterdir():
        dest = PROJECT_ROOT / f.name
        shutil.copy2(str(f), str(dest))
    
    return True, f"已从 {backup_id} 恢复"


def git_snapshot():
    """创建git快照作为额外保险"""
    try:
        subprocess.run(['git', 'add', '-A'], cwd=PROJECT_ROOT, 
                      capture_output=True, timeout=10)
        result = subprocess.run(
            ['git', 'commit', '--allow-empty', '-m', 
             f'COGNITION_BACKUP pre-{datetime.datetime.now().strftime("%H%M%S")}'],
            cwd=PROJECT_ROOT, capture_output=True, timeout=10, text=True
        )
        return result.stdout.strip()
    except Exception as e:
        return f"git backup failed: {e}"


def record_cognition(record, success=True):
    """写入project_cognition.md"""
    status = '✅ 已执行' if success else '❌ 失败'
    
    entry = f"""
### {record.cid}: {record.text[:50]}...
| 字段 | 值 |
|------|-----|
| **洞见** | {record.text} |
| **价值评分** | {record.value_score}/10 |
| **工时** | {record.effort_hours}h |
| **风险** | {record.risk_level} |
| **决策** | {record.decision} |
| **状态** | {status} |
| **文件** | {', '.join(record.files_changed)} |
| **时间** | {record.timestamp} |
| **回退** | `python cognition_pipeline.py rollback {record.cid}` |
"""
    with open(COGNITION_DB, 'a', encoding='utf-8') as f:
        f.write(entry)


# ============================================
# 3. CLI - 由AI/用户调用
# ============================================

def cmd_status():
    """查看所有认知状态"""
    print(f"\n认知数据库: {COGNITION_DB}")
    print(f"备份目录: {BACKUP_DIR}/")
    print(f"待办文件: {TODO_FILE}")
    
    # 统计
    with open(COGNITION_DB, 'r', encoding='utf-8') as f:
        content = f.read()
    
    executed = content.count('✅ 已执行')
    failed = content.count('❌ 失败')
    deferred = content.count('defer')
    rejected = content.count('reject')
    
    print(f"\n统计:")
    print(f"  已执行: {executed}")
    print(f"  失败: {failed}")
    print(f"  暂缓: {deferred}")
    print(f"  拒绝: {rejected}")
    
    # 最近的备份
    backups = sorted(BACKUP_DIR.iterdir()) if BACKUP_DIR.exists() else []
    if backups:
        print(f"\n最近备份: {backups[-1].name}")


def cmd_rollback(cid):
    """根据认知ID回退"""
    # 查找对应备份
    backup_id = None
    for b in sorted(BACKUP_DIR.iterdir()):
        if cid in b.name:
            backup_id = b.name
            break
    
    if backup_id:
        ok, msg = restore_backup(backup_id)
        print(f"{'✅' if ok else '❌'} {msg}")
    else:
        print(f"未找到 {cid} 的备份，尝试git恢复...")
        subprocess.run(['git', 'checkout', '--', '.'], cwd=PROJECT_ROOT)
        print("已git checkout恢复")


def cmd_rollback_last():
    """回退最近一次修改"""
    backups = sorted(BACKUP_DIR.iterdir()) if BACKUP_DIR.exists() else []
    if backups:
        cmd_rollback(backups[-1].name)
    else:
        print("没有可用备份")


if __name__ == '__main__':
    action = sys.argv[1] if len(sys.argv) > 1 else 'help'
    
    if action == 'status':
        cmd_status()
    elif action == 'rollback':
        if len(sys.argv) > 2:
            cmd_rollback(sys.argv[2])
        else:
            cmd_rollback_last()
    elif action in ('run', 'eval'):
        text = sys.argv[2] if len(sys.argv) > 2 else sys.stdin.read().strip()
        print(f"\n{'='*60}")
        print(f"🔍 认知评估请求")
        print(f"{'='*60}")
        print(f"洞见: {text}")
        print(f"\n--- AI评估阶段（以下由AI填写）---")
        print(f"动作: {'评估+执行' if action == 'run' else '仅评估'}")
        print(f"\n请AI评估此洞见并填充 EVAL_TEMPLATE 结构。")
        print(f"    run 模式: 评估后AI直接改代码")
        print(f"    eval 模式: 仅输出评估报告")
    else:
        print(f"""
认知评估与执行管道 v1

用法:
  python cognition_pipeline.py run "你的洞见"       评估+执行
  python cognition_pipeline.py eval "你的洞见"      仅评估不执行
  python cognition_pipeline.py status               查看认知状态
  python cognition_pipeline.py rollback <cid>        按认知ID回退
  python cognition_pipeline.py rollback              回退最近一次

工作流:
  1. AI评估洞见(价值/工时/风险)
  2. 输出评估报告
  3. 如果run模式: AI执行代码修改+备份+记录
  4. 如果eval模式: 仅输出报告

例:
  python cognition_pipeline.py run "SWA斜率应该替代深睡时长作为新指标"
""")
