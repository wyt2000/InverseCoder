import jsonlines

with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.no_python.evol-0325.python.summary') as f:
	instructions = [eval(line)['response'].strip() for line in f.readlines()]
with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.no_python.evol-0325.python') as f:
	responses = [eval(line)['response'].strip() for line in f.readlines()]

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

with jsonlines.open('magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.no_python.evol-0325.python.instruct', mode='w') as writer:
	for inst, resp in zip(instructions, responses):
		line = {
			'instruction' : inst,
			'response' : '```python\n' + extract_code(resp) + '\n```'
		}
		writer.write(line)
