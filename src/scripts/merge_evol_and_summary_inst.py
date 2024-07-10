import jsonlines
import ast

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl-generated-by-wizardcoder-gpt4-deepseekbase-0510-shiwenxuan') as f:
	responses = [redecode(eval(line)['response'].strip()) for line in f.readlines()]
with open('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl') as f:
	instructions = [redecode(eval(line)['instruction'].strip()) for line in f.readlines()]

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

with jsonlines.open('dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl-generated-by-wizardcoder-gpt4-deepseekbase-0510-shiwenxuan-resampled-0522', mode='w') as writer:	
    for i, resp in enumerate(responses):
        j = i // num_samples
        line = {
	    'instruction' : instructions[j],
	    'response' : resp
	}
        writer.write(line)
