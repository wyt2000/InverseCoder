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

raw_dataset = read_data('/lustre/S/wuyt/dataset/inverse_no_python/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.manual.bad-instructed-by-wizardcoder-gpt4-python-only-8x40g-0429-samples-10')

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

# dataset = sorted(raw_dataset, key=lambda x : x['yes_prob'])

num_samples = 10

def check(data):
    data['instruction'] = redecode(data['instruction'])
    data['response'] = redecode(data['response'])
    inst = data['instruction']
    if not inst: return False
    if 'recover' in inst or 'Recover' in inst: return False
    return True

good_dataset = []
bad_dataset = []
for i in range(0, len(raw_dataset), num_samples):
    for j in range(i, i + num_samples):
        data = raw_dataset[j]
        if check(data):
            good_dataset.append(data)
            break
    else:
        bad_dataset.append(raw_dataset[i])

#result_dataset = sorted(result_dataset, key=lambda x : x['yes_prob'])

with jsonlines.open('/lustre/S/wuyt/dataset/inverse_no_python/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.manual.bad-instructed-by-wizardcoder-gpt4-python-only-8x40g-0429-samples-10.good', mode='w') as writer:
    for data in good_dataset:
        writer.write(data)

with jsonlines.open('/lustre/S/wuyt/dataset/inverse_no_python/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.manual.bad-instructed-by-wizardcoder-gpt4-python-only-8x40g-0429-samples-10.bad', mode='w') as writer:
    for data in bad_dataset:
        writer.write(data)



