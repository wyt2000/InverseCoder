import jsonlines

with open('dataset/data-evol_instruct-decontaminated.jsonl.no_python-evol-by-deepseek-coder-6.7b-instruct-summarized-by-deepseek-coder-6.7b-instruct') as f:
	instructions = [eval(line)['response'].strip() for line in f.readlines()]
with open('dataset/data-evol_instruct-decontaminated.jsonl.no_python-evol-by-deepseek-coder-6.7b-instruct') as f:
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

with jsonlines.open('dataset/data-evol_instruct-decontaminated.jsonl.no_python-instruct-by-deepseekcoder-6.7b-instruct', mode='w') as writer:
	for inst, resp in zip(instructions, responses):
		line = {
			'instruction' : inst,
			'response' : resp
		}
		writer.write(line)
