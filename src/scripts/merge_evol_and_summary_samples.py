import jsonlines
import ast

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code') as f:
	responses = [redecode(eval(line)['response'].strip()) for line in f.readlines()]
with open('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code-summarized-by-wizardcoder-gpt4-QW-8x40g-0611') as f:
	instructions = [redecode(eval(line)['response'].strip()) for line in f.readlines()]

def extract_code(code: str):
    if not '```' in code:
        return code
    start = code.find('```')
    end = code.rfind('```')
    ret = code[start:end]
    # ret = '\n'.join(ret.splitlines()[1:])
    if not ret:
        ret = code[start:]
        # ret = '\n'.join(ret.splitlines()[1:])
    return ret

num_samples = 10

with jsonlines.open('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code-instructed-by-wizardcoder-gpt4-QW-8x40g-0611', mode='w') as writer:
    for i, inst in enumerate(instructions):
        j = i // num_samples
        line = {
	    'instruction' : inst,
	    'response' : responses[j]
	}
        writer.write(line)
