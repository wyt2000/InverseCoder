import random
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('dataset/mbpp/mbpp-prompt.jsonl') as f:
    lines = list(f.readlines())
    with jsonlines.open('dataset/mbpp/mbpp-prompt.jsonl.asserts', mode='w') as writer:
        for line in lines:
            line = eval(line)
            line['prompt'] = redecode(line['prompt'].split('Your code should satisfy the following assertion:\n')[1])
            writer.write(line)
