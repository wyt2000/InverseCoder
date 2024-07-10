import jsonlines
import ast

mbpp2ass = {}
with open('dataset/mbpp/mbpp-prompt.jsonl.asserts') as f:
    for line in f.readlines():
        line = eval(line)
        mbpp2ass[line['task_id']] = line['prompt']

mbpp2inst = {}
with open('test_results/mbpp-wizardcoder-gpt4-reproduce-0424-sanitized.jsonl-instructed-by-wizardcoder-gpt4-reproduce-0424') as f:
    for line in f.readlines():
        line = eval(line)
        task_id = line['task_id']
        mbpp2inst[task_id] = line['instruction'] + '\nYour code should satisfy the following assertion:\n' + mbpp2ass[task_id]

def extract_code(code: str):
    if not '```' in code:
        return code
    start = code.find('```')
    end = code.rfind('```')
    ret = code[start:end]
    if not ret:
        ret = code[start:]
    return ret

with jsonlines.open('test_results/mbpp-wizardcoder-gpt4-reproduce-0424-sanitized.jsonl-instructed-by-wizardcoder-gpt4-reproduce-0424-with-new-prompt', mode='w') as writer:	
    for task, inst in mbpp2inst.items():
        line = {
	    'task_id' : task,
	    'prompt' : inst 
	}
        writer.write(line)
