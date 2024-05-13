import json

def load_dataset(path):
    with open(path, 'r') as f:
        lines = f.readlines()
    lines = list(map(json.loads, lines))

    # Remove assistant (or anything that's not user) responses
    for line in lines:
        msgs = line['messages']
        if msgs[-1]['role'] != 'user':
            line['messages'] = msgs[:-1]
    
    return lines


def save_dataset(conversations, path):
    with open(path, 'w', encoding='utf-8') as f:
        for conv in conversations:
            f.write(json.dumps(conv, ensure_ascii=False) + '\n')