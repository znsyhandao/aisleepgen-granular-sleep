"""
认知注入管道 - 把科研/AI洞见自动转化为代码Change

用法:
  python cognition_inject.py <source> [--file <path>] [--manifest <doc>]

来源:
  text    : 从文本直接注入 ("SWA斜率比深睡时长更重要")
  paper   : 从论文摘要注入
  idea    : 自由想法注入

示例:
  python cognition_inject.py text "睡眠效率是旧范式，SWA斜率才是新目标" --manifest docs/manifesto.md
  
流程:
  1. 解析输入的认知
  2. 扫描项目代码，找"可注入点"（标记了 COGNITION_ANCHOR 的区域）
  3. 如果找到匹配锚点，更新代码
  4. 如果找不到匹配锚点，写入认知待办列表
  5. 更新 manifesto.md 或 project_cognition.md
"""

import sys
import os
import re
import json
import datetime

# 强制UTF-8输出(避免GBK编码错误)
sys.stdout.reconfigure(encoding='utf-8') if hasattr(sys.stdout, 'reconfigure') else None

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
COGNITION_FILE = os.path.join(PROJECT_ROOT, 'project_cognition.md')
ANCHOR_PATTERN = re.compile(r'COGNITION_ANCHOR\[([^\]]+)\]\s*{([^}]+)}', re.DOTALL)

# ============================================
# 锚点系统 - 在代码中标记可注入点
# ============================================
# 用法: 
#   # COGNITION_ANCHOR[认知ID]{旧假设}
#   # COGNITION_INJECT[认知ID]{新认知}
#   相关代码...

def find_anchors(paths=None):
    """扫描项目目录下所有带 COGNITION_ANCHOR 标记的代码"""
    if paths is None:
        paths = [
            os.path.join(PROJECT_ROOT, 'granular'),
            os.path.join(PROJECT_ROOT, 'experiments'),
        ]
    
    anchors = []
    for path in paths:
        if not os.path.isdir(path):
            continue
        for root, dirs, files in os.walk(path):
            for f in files:
                if f.endswith('.py'):
                    fp = os.path.join(root, f)
                    with open(fp, 'r', encoding='utf-8') as fh:
                        content = fh.read()
                    for m in ANCHOR_PATTERN.finditer(content):
                        anchors.append({
                            'id': m.group(1),
                            'old_assumption': m.group(2).strip(),
                            'file': fp,
                            'full_match': m.group(0),
                        })
    return anchors


def inject_cognition(cognition_text, source_type='text', manifest_path=None):
    """将认知注入项目"""
    ts = datetime.datetime.now().strftime('%Y-%m-%d %H:%M')
    
    # 1. 提取核心关键词作为锚点ID
    words = cognition_text.split()
    keywords = [w for w in words if len(w) > 2 and not w.startswith(('的','了','是','在','有','和','就','也','都','要'))]
    anchor_id = keywords[0] if keywords else 'unknown'
    
    # 2. 扫描现有锚点
    anchors = find_anchors()
    matching = [a for a in anchors if anchor_id.lower() in a['id'].lower() or any(k.lower() in a['id'].lower() for k in keywords[:3])]
    if not matching:
        _known_ids = {'SWA','narrative','pet','wormhole','presence'}
        for a in anchors:
            for kid in _known_ids:
                if kid.lower() in a['id'].lower() and kid.lower() in cognition_text.lower():
                    matching.append(a)
                    break
    
    result = {
        'timestamp': ts,
        'cognition': cognition_text,
        'source': source_type,
        'anchor_id': anchor_id,
        'matched_anchors': len(matching),
        'injected': False,
        'detail': ''
    }
    
    if matching:
        # 找到匹配锚点 → 注入代码
        for anchor in matching:
            anchor_file = anchor['file']
            with open(anchor_file, 'r', encoding='utf-8') as f:
                code = f.read()
            
            # 用新认知替换旧假设区域
            old = anchor['full_match']
            new = old.replace(
                anchor['old_assumption'],
                f'{anchor["old_assumption"]} | INJECTED[{ts}]: {cognition_text[:80]}'
            )
            code = code.replace(old, new)
            
            with open(anchor_file, 'w', encoding='utf-8') as f:
                f.write(code)
            
            result['injected'] = True
            result['detail'] += f"\n  注入: {anchor['file']} → {anchor['id']}"
    else:
        # 没有匹配锚点 → 写入认知待办
        result['detail'] = f'  无匹配锚点，写入认知待办'
        _append_cognition_todo(ts, cognition_text, anchor_id)
    
    # 3. 更新project_cognition.md
    _log_cognition(ts, cognition_text, result['injected'])
    
    return result


def _append_cognition_todo(ts, text, anchor_id):
    """写入认知待办列表"""
    todo_path = os.path.join(PROJECT_ROOT, 'cognition_todo.md')
    entry = (
        f"\n## {ts}\n"
        f"- **洞见**: {text}\n"
        f"- **建议锚点ID**: `{anchor_id}`\n"
        f"- **待做**: 在代码中添加 COGNITION_ANCHOR[{anchor_id}]{{旧假设}} 标记\n"
    )
    with open(todo_path, 'a', encoding='utf-8') as f:
        f.write(entry)
    print(f"[TODO] 写入 {todo_path}")


def _log_cognition(ts, text, injected):
    """记录认知到 project_cognition.md"""
    mode = '🧪 已注入代码' if injected else '📝 待办（无锚点）'
    entry = f"\n| {ts} | {mode} | {text[:60]}... |\n"
    
    if os.path.exists(COGNITION_FILE):
        with open(COGNITION_FILE, 'r', encoding='utf-8') as f:
            content = f.read()
        # 在表格后追加
        table_end = content.rfind('\n---')
        if table_end == -1:
            content += '\n\n## 输入历史\n| 时间 | 状态 | 认知 |\n|------|------|------|\n'
        with open(COGNITION_FILE, 'w', encoding='utf-8') as f:
            f.write(content + entry)
    else:
        with open(COGNITION_FILE, 'w', encoding='utf-8') as f:
            f.write(f'# 项目认知数据库\n\n## 输入历史\n| 时间 | 状态 | 认知 |\n|------|------|------|\n{entry}')


def init_project_cognition():
    """初始化项目的认知数据库（仅首次运行）"""
    if not os.path.exists(COGNITION_FILE):
        # 从现有的manifesto提取核心认知
        manifesto_path = os.path.join(PROJECT_ROOT, 'docs', 'manifesto.md')
        if os.path.exists(manifesto_path):
            with open(manifesto_path, 'r', encoding='utf-8') as f:
                manifesto = f.read()
        else:
            manifesto = ""
        
        content = f"""# 项目认知数据库

> 所有「读到好内容 → 注入到代码」的记录。

## 当前认知锚点

| 锚点ID | 旧假设 | 新认知 | 状态 |
|--------|--------|--------|------|
| `SWA` | 深睡时长最大化 | 突触重置效率最大化 | 🟢 已写入manifesto |
| `narrative` | 评分告诉用户答案 | 评分转化为叙事体验 | 🟢 已写入manifesto |
| `presence` | DAU最大化为目标 | 最小化存在感 | 🟢 已写入manifesto |
| `pet` | 文字/图表展示 | 界面宠物反馈 | 🟡 待锚点标记 |
| `wormhole` | 计时等待入睡 | 生成式时间膨胀 | 🔵 长期研究 |

## 注入历史
| 时间 | 状态 | 认知 |
|------|------|------|
"""
        with open(COGNITION_FILE, 'w', encoding='utf-8') as f:
            f.write(content)
        print(f"[INIT] 创建 {COGNITION_FILE}")
    else:
        print(f"[SKIP] {COGNITION_FILE} 已存在")


def generate_anchor_boilerplate(anchor_id):
    """生成代码锚点的样板"""
    return f"""
# COGNITION_ANCHOR[{anchor_id}]{{旧假设：请替换为当前假设}}
# 此区域标记了可被认知注入管道更新的代码。
# 当新的洞见与 {anchor_id} 相关时，cognition_inject.py 会自动替换 b 区域。
def handle_{anchor_id}(data):
    # TODO: 实现 {anchor_id} 的逻辑
    pass
"""


# ============================================
# CLI
# ============================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='认知注入管道')
    parser.add_argument('action', choices=['init', 'inject', 'scan', 'anchor'], 
                        help='init=初始化, inject=注入认知, scan=扫描锚点, anchor=生成锚点样板')
    parser.add_argument('source', nargs='?', help='认知文本来源')
    parser.add_argument('--file', help='从文件读取认知')
    parser.add_argument('--manifest', help='manifesto.md路径')
    parser.add_argument('--anchor-id', help='锚点ID（用于 anchor action）')
    
    args = parser.parse_args()
    
    if args.action == 'init':
        init_project_cognition()
    elif args.action == 'scan':
        anchors = find_anchors()
        print(f"\n找到 {len(anchors)} 个认知锚点:")
        for a in anchors:
            status = '✅ 已注入' if 'INJECTED' in a['old_assumption'] else '⏳ 待注入'
            print(f"  {status} {a['id']} → {os.path.relpath(a['file'], PROJECT_ROOT)}")
            print(f"    旧假设: {a['old_assumption'][:60]}")
    elif args.action == 'anchor':
        if not args.anchor_id:
            print("错误: 需要 --anchor-id")
            sys.exit(1)
        print(generate_anchor_boilerplate(args.anchor_id))
    elif args.action == 'inject':
        if args.file:
            with open(args.file, 'r', encoding='utf-8') as f:
                text = f.read()
        elif args.source:
            text = args.source
        else:
            text = sys.stdin.read()
        
        result = inject_cognition(text.strip(), manifest_path=args.manifest)
        print(f"\n{'='*50}")
        print(f"认知注入结果:")
        print(f"  时间: {result['timestamp']}")
        print(f"  状态: {'✅ 已注入代码' if result['injected'] else '📝 写入待办'}")
        print(f"  匹配锚点: {result['matched_anchors']}个")
        print(f"  详情: {result['detail']}")
