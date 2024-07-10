import json
import jsonlines

input_path = 'dataset/oss-instruct/oss-instruct.jsonl.code-instructed-by-magicoder-DS-reproduce-0501-select-by-magicoder-DS-reproduce-0501-with-score-sorted-with-problem-prefix'

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

dataset = read_data(input_path)
result_dataset = sorted(dataset, key=lambda x : len(x['response']))

with jsonlines.open(f'{input_path}-sorted-by-length', mode='w') as writer:
    for data in result_dataset:
        writer.write(data)


