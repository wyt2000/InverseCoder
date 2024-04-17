import random
import pandas as pd
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

data_path = 'dataset/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.all.shuffled.reshuffled'
with open(data_path) as f:
    lines = list(f.readlines())
    with jsonlines.open(f'{data_path}.cleaned', mode='w') as writer:
        for i, line in enumerate(lines):
            if 6145 <= i + 1 <= 7168: continue
            line = eval(line)
            line['instruction'] = redecode(line['instruction'])
            line['response'] = redecode(line['response'])
            writer.write(line)
