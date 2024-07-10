import random
import pandas as pd
import jsonlines
random.seed(42)

path = 'dataset/oss-instruct/oss-instruct.jsonl.code.fixed.sorted.removed.shortest.30-instructed-by-magicoder-CL-reproduce-0606-select-by-magicoder-CL-reproduce-0606-with-score-sorted'

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open(path) as f:
    lines = list(f.readlines())
    with jsonlines.open(f'{path}-with-problem-prefix', mode='w') as writer:
        for line in lines:
            line = eval(line)
            data = {}
            data['instruction'] = redecode(line['instruction'])
            data['instruction'] = 'Write a solution to the following coding problem:\n' + data['instruction'] 
            data['response'] = redecode(line['response'])
            writer.write(data)
