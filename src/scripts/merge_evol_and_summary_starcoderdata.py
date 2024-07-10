import jsonlines

with open('dataset/starcoderdata_cleaned_0314.jsonl') as f:
	responses = [eval(line)['raw_code'].strip() for line in f.readlines()]
with open('dataset/starcoderdata_cleaned_0314.jsonl-summarized-by-wizardcoder-gpt4') as f:
	instructions = [eval(line)['response'].strip() for line in f.readlines()]

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

with jsonlines.open('dataset/starcoderdata_cleaned_0314.jsonl-instructed-by-wizardcoder-gpt4', mode='w') as writer:
	for inst, resp in zip(instructions, responses):
		line = {
			'instruction' : inst,
			'response' : resp
		}
		writer.write(line)
