import random
import jsonlines
import ast
random.seed(42)

def is_begin_end(inst: str, resp: str):
    if 'BEGIN SOLUTION' in inst or 'BEGIN SOLUTION' in resp:
        return True 
    if 'END SOLUTION' in inst or 'END SOLUTION' in resp:
        return True 
    return False

def transform(code):
    code = code.replace('&lt;', '<')
    code = code.replace('&gt;', '>')
    return code

def convert_to_comment(code):
    lines = []
    for line in code.splitlines():
        x = line.strip()
        indent = len(line) - len(x)
        if x.startswith('BEGIN SOLUTION') or x.startswith('END SOLUTION'):
            lines.append(' ' * indent + '### ' + x)
        else:
            lines.append(line)
    return '\n'.join(lines)

def extract_code(inst, resp):
    start = inst.find('<code>') + len('<code>')
    end = inst.find('</code>', start + 1)
    if end != -1:
        inst = inst[start:end]
    else:
        inst = inst[start:]
    end = resp.find('</code>')
    if end != -1:
        resp = resp[:end]
    inst = convert_to_comment(inst)
    resp = convert_to_comment(resp)
    return inst, resp

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

path = 'dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.badcode'
with open(path) as f:
    target_dataset = []
    other_dataset = []

    lines = list(f.readlines())
    for line in lines:
        line = eval(line)
        inst = redecode(line['instruction'])
        resp = redecode(line['response'])
        if is_begin_end(inst, resp):
            inst = transform(inst)
            resp = transform(resp)
            inst, resp = extract_code(inst, resp)
            inst = inst.strip('\n')
            resp = resp.strip('\n')
            full_code = inst + '\n' + resp
            try:
                full_code = full_code.replace('&#39;', '\'')
                full_code = full_code.replace('&#34;', '\"')
                ast.parse(full_code)
                target_dataset.append({'response': '```python\n' + full_code + '```'})
            except Exception as err:
                print('@@code')
                print(full_code)
        else:
            other_dataset.append({'instruction': inst, 'response': resp})

    with jsonlines.open(f'{path}.begin_end', mode='w') as writer:
        for data in target_dataset:
            writer.write(data)
    with jsonlines.open(f'{path}.no_begin_end', mode='w') as writer:
        for data in other_dataset:
            writer.write(data)

