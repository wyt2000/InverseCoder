import random
import pandas as pd
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

def read_data(path):
    dataset = []
    with open(path) as f:
        lines = f.readlines()
        for line in lines:
            line = eval(line)
            line['instruction'] = redecode(line['instruction'])
            line['response'] = redecode(line['response'])
            dataset.append(line)
    return dataset

raw_dataset = read_data('/lustre/S/wuyt/dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.python-instructed-by-wizardcoder-gpt4')
scored_dataset = read_data('/lustre/S/wuyt/dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.python-instructed-by-wizardcoder-gpt4-select-by-wizardcoder-gpt4-modified-prompt-0426')

for x, y in zip(raw_dataset, scored_dataset):
    x['yes_prob'] = y['yes_prob']

def extract_code(code: str):
    if not '```' in code:
        return code
    start = code.find('```')
    end = code.rfind('```')
    ret = code[start:end]
    ret = '\n'.join(ret.splitlines()[1:])
    if not ret:
        ret = code[start:]
        ret = '\n'.join(ret.splitlines()[1:])
    return ret

def all_print(code):
    for line in code.splitlines():
        if not line.strip().startswith('print') and not line.strip().startswith('#'):
            return False 
    return True

dataset = sorted(raw_dataset, key=lambda x : x['yes_prob'])
#target_dataset = []
#for data in dataset[::-1]:
#    if len(target_dataset) == 10000: break
#    code = extract_code(data['response'])
#    if all_print(code): continue
#    target_dataset.append(data)

with jsonlines.open('/lustre/S/wuyt/dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.python-instructed-by-wizardcoder-gpt4-select-by-wizardcoder-gpt4-modified-prompt-0426-sorted', mode='w') as writer:
    for data in dataset:
        writer.write(data)


