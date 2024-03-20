import ast_comments as ast
import ast as old_ast
import jsonlines
import tqdm

class RemoveFuncImplTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.cnt = 0
    def visit_FunctionDef(self, node):
        doc = ast.get_docstring(node, clean=False)
        if node.body and isinstance(node.body[0], ast.Comment):
            node.body = [node.body[0]]
        else:
            node.body = []
        if doc:
            node.body.append(ast.Expr(value=ast.Str(doc)))
        node.body.append(ast.Pass())
        node.body.append(ast.Comment(value='# TODO', inline=True))
        self.cnt += 1
        return node

class RemoveClassImplTransformer(ast.NodeTransformer):
    def __init__(self):
        super().__init__()
        self.cnt = 0
    def visit_ClassDef(self, node):
        doc = ast.get_docstring(node, clean=False)
        node.body = []
        if doc:
            node.body.append(ast.Expr(value=ast.Str(doc)))
        node.body.append(ast.Pass())
        node.body.append(ast.Comment(value='# TODO', inline=True))
        self.cnt += 1
        return node

class CommentCollectVisitor(ast.NodeVisitor):
    def __init__(self):
        super().__init__()
        self.comments = []
    def visit_Comment(self, node):
        self.comments.append(node)

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
            if node.name == '__init__':
                continue
            if not _check_node(node):
                return False
    return True

class Unparser(ast._Unparser):
    def visit_Constant(self, node):
        # keep multi-line str unchanged
        if isinstance(node.value, str) and node.lineno < node.end_lineno:
            super()._write_str_avoiding_backslashes(node.value)
            return
        return super().visit_Constant(node)

    def newline_except_comment(self):
        if self._source:
            lastcode = self._source[-1].strip()
            if not lastcode.startswith('#') or lastcode.startswith('# TODO'):
                self.write("\n")

    def visit_Comment(self, node) -> None:
        if node.inline:
            self.write(f"  {node.value}")
        else:
            self.newline_except_comment()
            self.fill(node.value)

    def _function_helper(self, node, fill_suffix):
        self.newline_except_comment()
        for deco in node.decorator_list:
            self.fill("@")
            self.traverse(deco)
        def_str = fill_suffix + " " + node.name
        self.fill(def_str)
        if hasattr(node, "type_params"):
            self._type_params_helper(node.type_params)
        with self.delimit("(", ")"):
            self.traverse(node.args)
        if node.returns:
            self.write(" -> ")
            self.traverse(node.returns)
        with self.block(extra=self.get_type_comment(node)):
            self._write_docstring_and_traverse_body(node)

    def visit_ClassDef(self, node):
        self.newline_except_comment()
        for deco in node.decorator_list:
            self.fill("@")
            self.traverse(deco)
        self.fill("class " + node.name)
        if hasattr(node, "type_params"):
            self._type_params_helper(node.type_params)
        with self.delimit_if("(", ")", condition = node.bases or node.keywords):
            comma = False
            for e in node.bases:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.traverse(e)
            for e in node.keywords:
                if comma:
                    self.write(", ")
                else:
                    comma = True
                self.traverse(e)

        with self.block():
            self._write_docstring_and_traverse_body(node)


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
    if visitor.cnt:
        return unparse(root)
    visitor = RemoveClassImplTransformer()
    visitor.visit(root)
    if visitor.cnt:
        return unparse(root)
    visitor = CommentCollectVisitor()
    visitor.visit(root)
    doc = ast.get_docstring(root, clean=False)
    if doc: root.body = [ast.Expr(value=ast.Str(doc))]
    else: root.body = []
    root.body.extend(visitor.comments)
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
with open('magicoder_data/starcoderdata_cleaned_0314_with_comment_0319.jsonl') as f:
    lines = list(f.readlines())
    with tqdm.tqdm(total=len(lines)) as pbar:
        for line in lines:
            line = eval(line)
            code = extract_code(line['response'])
            try:
                if not code:
                    print(line['response'])
                    raise ValueError('Empty code!')
                if not check_code_implementation(code):
                    # print(line['response'])
                    raise ValueError('No implementation!')
                prompt = remove_impl(code)
                old_ast.parse(prompt)
                prompt = code_templete.format(code=prompt)
                prompt = prompt_templete.format(prompt=prompt)
                code = clean_code(code)
                old_ast.parse(code)
                data = {'instruction': prompt, 'response': code_templete.format(code=code)}
                dataset.append(data)
            except Exception as err:
                #print(line['response'])
                #print(code)
                #print(err)
                pass
            finally:
                pbar.update(1)

with jsonlines.open('magicoder_data/starcoderdata_cleaned_0314_instructed_by_comment_0319.jsonl', mode='w') as writer:
    for data in dataset:
        writer.write(data)


