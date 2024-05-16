import random
import jsonlines
import ast
random.seed(42)

def is_sql(inst: str, resp: str):
    if '|:-' in inst or '|:-' in resp:
        return True 
    return False

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

path = 'dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.badcode.no_begin_end'
with open(path) as f:
    target_dataset = []
    other_dataset = []

    lines = list(f.readlines())
    for line in lines:
        line = eval(line)
        inst = redecode(line['instruction'])
        resp = redecode(line['response'])
        if is_sql(inst, resp):
            target_dataset.append({'instruction': inst, 'response': resp})
        else:
            other_dataset.append({'instruction': inst, 'response': resp})

    with jsonlines.open(f'{path}.sql', mode='w') as writer:
        for data in target_dataset:
            writer.write(data)
    with jsonlines.open(f'{path}.no_sql', mode='w') as writer:
        for data in other_dataset:
            writer.write(data)

