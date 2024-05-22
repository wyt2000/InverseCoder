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

raw_dataset = read_data('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl-generated-by-wizardcoder-gpt4-deepseekbase-0510-shiwenxuan-resampled-0522')
scored_dataset = read_data('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl-generated-by-wizardcoder-gpt4-deepseekbase-0510-shiwenxuan-resampled-0522-select-by-wizardcoder-gpt4-deepseekbase-0510-shiwenxuan')

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

# dataset = sorted(raw_dataset, key=lambda x : x['yes_prob'])

num_samples = 10

result_dataset = []
for i in range(0, len(raw_dataset), num_samples):
    data = sorted(raw_dataset[i : i + num_samples], key=lambda x : -x['yes_prob'])[0]
    result_dataset.append(data)
result_dataset = sorted(result_dataset, key=lambda x : x['yes_prob'])

with jsonlines.open('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl-generated-by-wizardcoder-gpt4-deepseekbase-0510-shiwenxuan-resampled-0522-select-by-wizardcoder-gpt4-deepseekbase-0510-shiwenxuan-with-score-sorted', mode='w') as writer:
    for data in result_dataset:
        writer.write(data)



