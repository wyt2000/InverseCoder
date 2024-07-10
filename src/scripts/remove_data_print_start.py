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

#test_code = '\nimport boto3 # Error: This library does not exist in Python\nfrom sagemaker import Session # Error: This library does not exist in Python\n\n# Here your code would go...\n'''

def all_print(code):
    for line in code.splitlines():
        line = line.strip()
        if line and not line.startswith('print') and not line.startswith('#') and not line.startswith('import') and not line.startswith('from'):
            return False 
    return True

#all_print(test_code)
#exit(0)

path = 'dataset/inverse_no_python/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct-instructed-by-wizardcoder-gpt4-python-only-8x40g-0429-problem-prompt-samples-10-select-by-wizardcoder-gpt4-reproduce-0424-with-score-sorted'
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


