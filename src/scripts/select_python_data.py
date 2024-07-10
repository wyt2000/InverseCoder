import random
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

def check(data):
    data['instruction'] = redecode(data['instruction'])
    data['response'] = redecode(data['response'])
    inst = data['instruction']
    resp = data['response']
    start = inst.find(' a')
    inst = inst[start + 2:].strip()
    return inst.startswith('py') or inst.startswith('Py')

path = '/lustre/S/wuyt/dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code-instructed-by-wizardcoder-gpt4-deepseekbase-reproduce-4x80g-0505-problem-prompt-samples-10-select-by-wizardcoder-gpt4-deepseekbase-reproduce-4x80g-0505-with-score-sorted'

with open(path) as f:
    lines = list(f.readlines())
    dataset = []
    for line in lines:
        line = eval(line)
        if check(line):
            dataset.append({'instruction': line['instruction'], 'response': line['response']})
    with jsonlines.open(f'{path}.python', mode='w') as writer:
        for line in dataset:
            writer.write(line)

