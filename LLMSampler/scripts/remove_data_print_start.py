import random
import pandas as pd
import jsonlines
random.seed(42)

def extract_code(code: str):
    if not '```' in code:
        return code
    start = code.find('```')
    end = code.rfind('```')
    ret = code[start:end]
    ret = '\n'.join(ret.splitlines()[1:])
    if not ret:
        ret = code[start:]
        ret = '\n'.join(ret.splitlines()[1:])
    return ret

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

def all_print(code):
    for line in code.splitlines():
        if not line.strip().startswith('print') and not line.strip().startswith('#'):
            return False 
    return True

path = 'dataset/AutoMathTextData-10k.jsonl.with.all.python.shuffled'
with open(path) as f:
    lines = list(f.readlines())
    with jsonlines.open(f'{path}.remove_all_print', mode='w') as writer:
        for line in lines:
            line = eval(line)
            line['instruction'] = redecode(line['instruction'])
            line['response'] = redecode(line['response'])
            code = extract_code(line['response'])
            if all_print(code): continue
            writer.write(line)


