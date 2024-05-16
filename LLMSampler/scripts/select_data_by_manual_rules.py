import random
import pandas as pd
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

def check(data):
    data['instruction'] = redecode(data['instruction'])
    data['response'] = redecode(data['response'])
    inst = data['instruction']
    if not inst: return False
    if 'recover' in inst or 'Recover' in inst: return False
    return True

with open('dataset/inverse_no_python/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct') as f:
    lines = list(f.readlines())
    good_dataset = []
    bad_dataset = []
    for line in lines:
        line = eval(line)
        if check(line):
            good_dataset.append(line)
        else:
            bad_dataset.append(line)
    with jsonlines.open('dataset/inverse_no_python/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.manual.good', mode='w') as writer:
        for line in good_dataset:
            writer.write(line)
    with jsonlines.open('dataset/inverse_no_python/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.manual.bad', mode='w') as writer:
        for line in bad_dataset:
            writer.write(line)
