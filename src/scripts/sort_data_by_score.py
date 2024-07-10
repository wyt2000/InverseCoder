import random
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

path = 'dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code-instructed-by-wizardcoder-gpt4-deepseekbase-reproduce-4x80g-0505-problem-prompt-samples-10-select-by-wizardcoder-gpt4-deepseekbase-reproduce-4x80g-0505-with-score-sorted-with-ds1000-like-data'
dataset = read_data(path)
dataset = sorted(dataset, key=lambda x : x['yes_prob'])

with jsonlines.open(f'{path}-sorted', mode='w') as writer:
    for data in dataset:
        writer.write(data)
