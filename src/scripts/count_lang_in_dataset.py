import random
import jsonlines
from collections import Counter
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

def get_lang(data):
    data['instruction'] = redecode(data['instruction'])
    data['response'] = redecode(data['response'])
    inst = data['instruction']
    resp = data['response']
    start = inst.find(' a')
    inst = inst[start + 2:].strip()
    lang = inst.split(' ')[0]
    if lang == 'rust':
        print('@@inst')
        print(inst)
        print('@@resp')
        print(resp)
    return lang

path = '/lustre/S/wuyt/dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code-instructed-by-wizardcoder-gpt4-reproduce-0424-problem-prompt-samples-10-select-by-wizardcoder-gpt4-reproduce-0424-with-score-sorted'

cnt = Counter()
with open(path) as f:
    lines = list(f.readlines())
    dataset = []
    for line in lines:
        line = eval(line)
        lang = get_lang(line)
        cnt[lang] += 1
print(cnt)

