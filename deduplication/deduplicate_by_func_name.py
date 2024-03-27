import tqdm
import jsonlines
import ast
dataset = []
with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.no_python.all.instruct.0326.shuffled') as f:
    for line in f.readlines():
        line = eval(line)
        dataset.append(line)

class FuncNameVisitor(ast.NodeVisitor):
    def __init__(self):
        self.func_names = []
    def visit_FunctionDef(self, node):
        self.generic_visit(node)
        self.func_names.append(node.name)

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
func_names_table = set()
deduplicated_dataset = []

with tqdm.tqdm(total=len(dataset)) as pbar:
    for data in dataset:
        try:
            code = extract_code(data['response'])
            root = ast.parse(code)
        except Exception as err:
            continue
        visitor = FuncNameVisitor()
        visitor.visit(root)
        func_names = tuple(set(visitor.func_names))
        if len(func_names) == 0:
            pass
        elif len(func_names) == 1 and (len(func_names[0]) == 1 or func_names[0] == 'solve' or func_names[0] == 'solution'):
            pass
        elif func_names in func_names_table:
            continue
        func_names_table.add(func_names)
        deduplicated_dataset.append(data)
        pbar.update(1)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with jsonlines.open('magicoder_data/data-evol_instruct-decontaminated.jsonl.no_python.all.instruct.0326.shuffled.deduplicated', mode='w') as writer:
    for data in deduplicated_dataset:
        data['instruction'] = redecode(data['instruction'])
        data['response'] = redecode(data['response'])