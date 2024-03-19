import ast_comments as ast
import jsonlines
import tqdm

class RemoveFuncImplTransformer(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        doc = ast.get_docstring(node)
        node.body = []
        if doc:
            node.body.append(ast.Expr(value=ast.Str(doc)))
        node.body.append(ast.Pass())
        return node

class Unparser(ast._Unparser):
   def visit_Constant(self, node):
      if isinstance(node.value, str) and node.lineno < node.end_lineno:
         super()._write_str_avoiding_backslashes(node.value)
         return
      return super().visit_Constant(node)

def unparse(ast_node):
   u = Unparser()
   return u.visit(ast_node)

def clean_code(code):
    root = ast.parse(code)
    return unparse(root)

def remove_impl(code):
    root = ast.parse(code)
    visitor = RemoveFuncImplTransformer()
    visitor.visit(root)
    return unparse(root)

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

code_templete = '''```python
{code}
```'''
prompt_templete = '''Please complete the code:
{prompt}'''

dataset = []
with open('magicoder_data/starcoderdata_cleaned_0314_with_docstring.jsonl') as f:
    lines = list(f.readlines())
    with tqdm.tqdm(total=len(lines)) as pbar:
        for line in lines:
            line = eval(line)
            code = extract_code(line['response'])
            try:
                if not code:
                    print(line['response'])
                    raise ValueError('Empty code!')
                prompt = remove_impl(code)
                prompt = code_templete.format(code=prompt)
                prompt = prompt_templete.format(prompt=prompt)
                code = clean_code(code)
                data = {'instruction': prompt, 'response': code_templete.format(code=code)}
                dataset.append(data)
            except Exception as err:
                #print(line['response'])
                #print(code)
                #print(err)
                pass
            finally:
                pbar.update(1)

with jsonlines.open('magicoder_data/starcoderdata_cleaned_0314_instructed_by_docstring_0319.jsonl', mode='w') as writer:
    for data in dataset:
        writer.write(data)


