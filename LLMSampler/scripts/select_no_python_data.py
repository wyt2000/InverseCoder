import random
import jsonlines
from collections import Counter
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

ps = [
    'java',
    'javascript',
    'typescript',
    'js',
    'ts',
    'cpp',
    'c++',
    'rust',
    'rs',
    'php',
    'swift',
    'csharp',
    'c#',
    'c ',
]

cnt = Counter()

def check(data):
    data['instruction'] = redecode(data['instruction'])
    data['response'] = redecode(data['response'])
    inst = data['instruction']
    resp = data['response']
    start = inst.find(' a')
    inst = inst[start + 2:].strip().lower()
    global cnt
    cnt[inst.split()[0]] += 1
    for p in ps:
        if inst.startswith(p):
            return True
    return False

path = '/lustre/S/wuyt/dataset/oss-instruct/oss-instruct.jsonl.code-instructed-by-magicoder-DS-reproduce-0501-select-by-magicoder-DS-reproduce-0501-with-score-sorted'

with open(path) as f:
    lines = list(f.readlines())
    dataset = []
    for line in lines:
        line = eval(line)
        if check(line):
            dataset.append({'instruction': line['instruction'], 'response': line['response']})
    with jsonlines.open(f'{path}.no_python', mode='w') as writer:
        for line in dataset:
            writer.write(line)

print(cnt)
