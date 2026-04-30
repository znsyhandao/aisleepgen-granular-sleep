import subprocess, os

cwd = r'D:\AISleepGen_GranularSleep'
cmds = [
    ['git', 'init', '-b', 'main'],
    ['git', 'config', 'user.email', 'bot@granular.sleep'],
    ['git', 'config', 'user.name', 'znsyhandao'],
    ['git', 'add', '-A'],
    ['git', 'commit', '-m', 'init: 粒计算睡眠阶段识别框架'],
    ['git', 'remote', 'add', 'origin', 'git@github.com:znsyhandao/aisleepgen-granular-sleep.git'],
]
for cmd in cmds:
    r = subprocess.run(cmd, capture_output=True, text=True, cwd=cwd)
    cmd_str = ' '.join(cmd)
    print('$ ' + cmd_str)
    out = (r.stdout or '').strip()
    if out:
        print('  ' + out[:200])
    err = (r.stderr or '').strip()
    if r.returncode != 0 and 'nothing to commit' not in err:
        print('  ERR: ' + err[:200])

# Push (separate to handle auth)
r = subprocess.run(['git', 'push', '-u', 'origin', 'main'], cwd=cwd, capture_output=True, text=True)
print('--- PUSH ---')
print((r.stdout or '').strip()[:500])
if r.returncode != 0:
    print('ERR: ' + (r.stderr or '').strip()[:300])
