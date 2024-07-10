import random
import jsonlines
import ast
random.seed(42)

def extract_code(inst: str, code: str):
    if not '```' in code:
        try:
            ast.parse(code)
        except Exception as err:
            raise Exception('No Code!')
        return '```python\n' + code + '```'

    start = code.find('```')
    end = code.rfind('```')
    ret = code[start:end]
    if not ret:
        ret = code[start:]
    ret += '```'
    return ret

def extract_code_first_block(inst: str, code: str):
    cnt = code.count('```')

    if cnt == 0:
        try:
            ast.parse(code)
        except Exception as err:
            raise Exception('No Code!')
        return '```python\n' + code + '```'

    target = ''
    if cnt == 1:
        x, y = code.split('```')
        if len(x) > len(y):
            target = x
        else:
            target = y
    else:
        target = code.split('```')[1]

    return '```' + target + '```'


def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')


bad_codes = []
path = 'dataset/oss-instruct/oss-instruct.jsonl'
with open(path) as f:
    lines = list(f.readlines())
    with jsonlines.open(f'{path}.code.fixed', mode='w') as writer:
        for line in lines:
            line = eval(line)
            inst = redecode(line['instruction'])
            resp = redecode(line['response'])
            try:
                code = extract_code_first_block(inst, resp)
                writer.write({'instruction': inst, 'response': code})
            except Exception as err:
                pass

    with jsonlines.open(f'{path}.badcode.fixed', mode='w') as writer:
        for line in lines:
            line = eval(line)
            inst = redecode(line['instruction'])
            resp = redecode(line['response'])
            try:
                code = extract_code_first_block(inst, resp)
            except Exception as err:
                writer.write({'instruction': inst, 'response': resp})
