import random
import pandas as pd
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('/lustre/S/wuyt/dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code-instructed-by-wizardcoder-gpt4-reproduce-0424-problem-prompt-samples-10-select-by-wizardcoder-gpt4-reproduce-0424-with-score-sorted-with-ds1000-like-data') as f:
    lines = list(f.readlines())
    random.shuffle(lines)
    with jsonlines.open('/lustre/S/wuyt/dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code-instructed-by-wizardcoder-gpt4-reproduce-0424-problem-prompt-samples-10-select-by-wizardcoder-gpt4-reproduce-0424-with-score-sorted-with-ds1000-like-data-shuffled', mode='w') as writer:
        for line in lines:
            line = eval(line)
            data = {}
            data['instruction'] = redecode(line['instruction'])
            data['response'] = redecode(line['response'])
            writer.write(data)
