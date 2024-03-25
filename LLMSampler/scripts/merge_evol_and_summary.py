import jsonlines

with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.python.summary-0324') as f:
	instructions = [eval(line)['response'].strip() for line in f.readlines()]
with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.python.instruct-0324.regenerate') as f:
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

with jsonlines.open('magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.python.regenerate.instruct-0325', mode='w') as writer:
	for inst, resp in zip(instructions, responses):
		line = {
			'instruction' : inst,
			'response' : resp
		}
		writer.write(line)
