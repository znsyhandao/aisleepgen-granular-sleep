def parse_sleep_stages(annotation):
    """解析睡眠阶段注释"""
    stage_mapping = {
        'W': 'Wake',
        '1': 'N1',
        '2': 'N2',
        '3': 'N3',
        '4': 'N3',  # 合并N3和N4
        'R': 'REM',
        '?': 'Unknown'
    }
    
    stages = []
    for symbol in annotation.symbol:
        stages.append(stage_mapping.get(symbol, 'Unknown'))
    
    return np.array(stages)
