import jsonlines                                                                                       

with open('magicoder_data/codevol-wizardcoder-0304-summaried-by-wizardcoder-reversed-problem-prompt.jsonl') as f:
	instructions = [eval(line)['response'].strip() for line in f.readlines()]
with open('magicoder_data/codevol-wizardcoder-0304.jsonl') as f:
	responses = [eval(line)['response'].strip() for line in f.readlines()]

with jsonlines.open('magicoder_data/codevol-wizardcoder-0304-instructed-by-wizardcoder-reversed-problem-prompt-0318.jsonl', mode='w') as writer:
	for inst, resp in zip(instructions, responses):
		line = {
			'instruction' : inst.strip(),
			'response' : resp.strip()
		}
		writer.write(line)
