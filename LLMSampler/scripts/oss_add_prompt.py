import random
import pandas as pd
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('dataset/oss-instruct/oss-instruct.jsonl.code-instructed-by-magicoder-CL-reproduce-2x80g-1100token-no-flashattn-0512-select-by-magicoder-CL-reproduce-2x80g-1100token-no-flashattn-0512-with-score-sorted') as f:
    lines = list(f.readlines())
    with jsonlines.open('dataset/oss-instruct/oss-instruct.jsonl.code-instructed-by-magicoder-CL-reproduce-2x80g-1100token-no-flashattn-0512-select-by-magicoder-CL-reproduce-2x80g-1100token-no-flashattn-0512-with-score-sorted-preprocessed', mode='w') as writer:
        for line in lines:
            line = eval(line)
            data = {}
            data['instruction'] = redecode(line['instruction'])
            data['instruction'] = 'Write a solution to the following coding problem:\n' + data['instruction'] 
            data['response'] = redecode(line['response'])
            writer.write(data)
