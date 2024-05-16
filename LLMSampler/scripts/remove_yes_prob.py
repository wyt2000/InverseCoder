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
            inst = redecode(line['instruction'])
            resp = redecode(line['response'])
            dataset.append({'instruction': inst, 'response': resp})
    return dataset

path = 'dataset/oss-instruct/oss-instruct.jsonl.code-instructed-by-wizardcoder-gpt4-reproduce-0424-problem-prompt-samples-10-select-by-wizardcoder-gpt4-reproduce-0424-with-score-sorted'

dataset = read_data(path)

with jsonlines.open(f'{path}.remove_score', mode='w') as writer:
    for data in dataset:
        writer.write(data)


