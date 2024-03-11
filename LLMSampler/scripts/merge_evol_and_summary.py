import jsonlines                                                                                       

with open('magicoder_data/oss-instruct-codevol-reversed-wizardcoder-gpt4-summary-0310.jsonl') as f:
	instructions = [eval(line)['response'].strip() for line in f.readlines()]
with open('magicoder_data/oss-instruct-codevol-0306.jsonl') as f:
	responses = [eval(line)['response'].strip() for line in f.readlines()]

with jsonlines.open('magicoder_data/oss-instruct-codevol-0306-summary-by-wizardcoder-3200-0310.jsonl', mode='w') as writer:
	for inst, resp in zip(instructions, responses):
		line = {
			'instruction' : inst,
			'response' : resp
		}
		writer.write(line)
