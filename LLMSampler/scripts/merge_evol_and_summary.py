import jsonlines

with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.no_python.summary-0327') as f:
	instructions = [eval(line)['response'].strip() for line in f.readlines()]
with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327') as f:
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

with jsonlines.open('magicoder_data/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct', mode='w') as writer:
	for inst, resp in zip(instructions, responses):
		line = {
			'instruction' : inst,
			'response' : resp
		}
		writer.write(line)
