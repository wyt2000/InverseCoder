import ast_comments as ast 

testcase = '''
# First question related code:
"""
This function calculates whether a number is a perfect cube or not.
The number to be checked is taken as a command-line argument.
If it's a perfect cube, then it returns True otherwise False.
"""
def perfect_cube(n):
    """
    Here is a docstring.
    """
    i = n // 100
    j = n // 10 % 10
    k = n % 10
    return n == i ** 3 + j ** 3 + k ** 3 

# The following code checks if the numbers between 400 and 500 are perfect cubes.
for n in range(400,500):
    if perfect_cube(n):
       print(n)


# Second question related code:
"""
This function takes a string of space separated digits and appends only those which 
are divisible by 6 into another list. It also prints these numbers.
"""
def divisible_by_six(s):
    l2 = []
    for i in s.split(' '):
        if i.isdigit() and int(i) % 6 == 0:
            l2.append(int(i))
    for i in l2:
        print(i, end=" ")

# To use it
divisible_by_six(input("Enter a list of integers separated by spaces: "))
'''

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

root = ast.parse(testcase)
visitor = RemoveFuncImplTransformer()
visitor.visit(root)
code = unparse(root)
print(testcase)
print(code)
