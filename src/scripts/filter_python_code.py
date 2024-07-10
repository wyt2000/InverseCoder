import ast
import jsonlines
import tqdm

def _check_node(node):
    if not node.body:
        return False
    if len(node.body) > 2:
        return True
    return not all(isinstance(x, (ast.Pass, ast.Str, ast.Raise)) for x in node.body)

def check_code_implementation(code):
    root = ast.parse(code)
    for node in ast.walk(root):
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if not _check_node(node):
                return False
    return True

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

input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl'
dataset = []
with open(input_path) as f:
    lines = list(f.readlines())
    with tqdm.tqdm(total=len(lines)) as pbar:
        for line in lines:
            line = eval(line)
            code = extract_code(line['response'])
            try:
                if not code:
                    print(line['response'])
                    raise ValueError('Empty code!')
                # ast.parse(code)
                if not check_code_implementation(code):
                    print(line['response'])
                    raise ValueError('No implementation!')
                data = {'instruction': redecode(line['instruction']), 'response': redecode(line['response'])}
                dataset.append(data)
            except Exception as err:
                #print(line['response'])
                #print(code)
                #print(err)
                pass
            finally:
                pbar.update(1)

print(len(dataset))
with jsonlines.open(f'{input_path}.python.completed', mode='w') as writer:
    for data in dataset:
        writer.write(data)


