import random
import jsonlines
import ast
random.seed(42)

def extract_code(inst: str, code: str):
    if not '```' in code:
        raise Exception('No Code!')
    start = code.find('```')
    end = code.rfind('```')
    ret = code[start:end]
    if not ret:
        ret = code[start:]
    ret += '```'
    return ret

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

bad_codes = []
path = 'dataset/oss-instruct/oss-instruct.jsonl'
with open(path) as f:
    lines = list(f.readlines())
    with jsonlines.open(f'{path}.code.with.old.prompt.compare', mode='w') as writer:
        for line in lines:
            line = eval(line)
            inst = redecode(line['instruction'])
            resp = redecode(line['response'])
            try:
                code = extract_code(inst, resp)
                writer.write({'response': code})
            except Exception as err:
                pass
    with jsonlines.open(f'{path}.badcode', mode='w') as writer:
        for line in lines:
            line = eval(line)
            inst = redecode(line['instruction'])
            resp = redecode(line['response'])
            try:
                code = extract_code(inst, resp)
            except Exception as err:
                # print(err)
                writer.write({'instruction': inst, 'response': resp})


