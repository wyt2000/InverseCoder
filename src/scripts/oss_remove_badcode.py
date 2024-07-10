import random
import pandas as pd
import jsonlines
random.seed(42)

path = 'dataset/oss-instruct/oss-instruct.jsonl.code-instructed-by-magicoder-DS-reproduce-0501-select-by-magicoder-DS-reproduce-0501-with-score-sorted-with-problem-prefix'

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open(path) as f:
    lines = list(f.readlines())
    with jsonlines.open(f'{path}-removed-badcode', mode='w') as writer:
        for line in lines:
            line = eval(line)
            data = {}
            data['instruction'] = redecode(line['instruction'])
            data['response'] = redecode(line['response'])
            if '`````' in data['response']: continue
            writer.write(data)
