import jsonlines
import ast

with open('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code') as f:
	responses = [eval(line)['response'].strip() for line in f.readlines()]
with open('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code-summarized-by-wizardcoder-gpt4-QW-8x40g-0611') as f:
	instructions = [eval(line)['response'].strip() for line in f.readlines()]

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

with jsonlines.open('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code-instructed-by-wizardcoder-gpt4-QW-8x40g-0611', 'w') as writer:	
    for inst, resp in zip(instructions, responses):
        code = resp
#        code = extract_code(resp)
        line = {
	    'instruction' : inst,
#	    'response' : '```python\n' + code + '```'
	    'response' : code 
	}
        writer.write(line)
