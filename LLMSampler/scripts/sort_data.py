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

raw_dataset = read_data('dataset/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.all.shuffled')
scored_dataset = read_data('dataset/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.all.shuffled-select-by-wizardcoder-gpt4')

for x, y in zip(raw_dataset, scored_dataset):
    x['yes_prob'] = y['yes_prob']

dataset = sorted(raw_dataset, key=lambda x : x['yes_prob'])
with jsonlines.open('dataset/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.all.shuffled-select-by-wizardcoder-gpt4.sorted.cleaned-20k', mode='w') as writer:
    for data in dataset[20000:]:
        if data['yes_prob'] == 1.0: continue
        writer.write(data)


