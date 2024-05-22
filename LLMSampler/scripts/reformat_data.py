import random
import json
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('/lustre/S/wuyt/dataset/evol-instruct-gpt35/EvolInstruct-Code-80k.json') as f:
    dataset = json.loads(f.read())
    with jsonlines.open('/lustre/S/wuyt/dataset/evol-instruct-gpt35/EvolInstruct-Code-80k-inst-data.jsonl', mode='w') as writer:
        for line in dataset:
            data = {}
            data['instruction'] = redecode(line['instruction'])
            data['response'] = redecode(line['output'])
            writer.write(data)
