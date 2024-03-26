import ast
import jsonlines
import tqdm
from collections import Counter

func_names = Counter()
benchmark_data = []
with open('magicoder_data/humaneval-benchmark.jsonl') as f:
    for line in f.readlines():
        line = eval(line)
        code = line['prompt']
        root = ast.parse(code)
        for node in ast.walk(root):
            if isinstance(node, ast.FunctionDef):
                func_names[node.name] = 0

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

data_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl'
fixed_dataset = []
with jsonlines.open(f'{data_path}.python', mode='w') as writer:
    with open(data_path) as f:
        dataset = list(f.readlines())
        with tqdm.tqdm(total=len(dataset)) as pbar:
            for line in dataset:
                line = eval(line)
                data = {
                    'instruction': redecode(line['instruction']),
                    'response': redecode(line['response']
                )}
                inst = line['instruction']
                resp = line['response']
                code = extract_code(resp)
                try:
                    if not code:
                        raise ValueError('Empty code!')
                    ast.parse(code)
                    writer.write(data)
                except Exception as err:
                    fixed_dataset.append(data)                    
                pbar.update(1)
with jsonlines.open(f'{data_path}.no_python', mode='w') as writer:
    for data in fixed_dataset:
        writer.write(data)

print(func_names)
print(sum(1 for value in func_names.values() if value > 0))
print(len(func_names))

