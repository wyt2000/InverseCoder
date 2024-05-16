import random
import pandas as pd
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('dataset/oss-instruct/oss-instruct.jsonl.code-instructed-by-magicoder-DS-reproduce-0501-select-by-magicoder-DS-reproduce-0501-with-score-sorted') as f:
    lines = list(f.readlines())
    good_dataset = []
    bad_dataset = []
    for line in lines:
        line = eval(line)
        if line['yes_prob'] >= 0.7:
            good_dataset.append(line)
        else:
            bad_dataset.append(line)
    with jsonlines.open('dataset/oss-instruct/oss-instruct.jsonl.code-instructed-by-magicoder-DS-reproduce-0501-select-by-magicoder-DS-reproduce-0501-with-score-sorted.good.70', mode='w') as writer:
        for line in good_dataset:
            writer.write(line)

