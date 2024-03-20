import random
import pandas as pd
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('magicoder_data/oss-instruct-codevol-0306-problem-prompt-instruction-data-cleaned.jsonl') as f:
    lines = list(f.readlines())
    sample_lines = random.sample(lines, len(lines) - 1648)
    with jsonlines.open('magicoder_data/oss-instruct-codevol-0306-problem-prompt-instruction-data-cleaned-random-remove-1648.jsonl', mode='w') as writer:
        for line in sample_lines:
            line = eval(line)
            line['instruction'] = redecode(line['instruction'])
            line['response'] = redecode(line['response'])
            writer.write(line)