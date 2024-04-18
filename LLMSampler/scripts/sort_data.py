import random
import pandas as pd
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('dataset/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.all.shuffled-select-by-wizardcoder-gpt4') as f:
    lines = list(f.readlines())
    random.shuffle(lines)
    dataset = []
    for line in lines:
        line = eval(line)
        line['instruction'] = redecode(line['instruction'])
        line['response'] = redecode(line['response'])
        dataset.append(line)
    dataset = sorted(dataset, key=lambda x : x['yes_prob'])
    with jsonlines.open('dataset/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.all.shuffled-select-by-wizardcoder-gpt4.sorted', mode='w') as writer:
        for data in dataset:
            writer.write(data)

