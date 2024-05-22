import jsonlines
import ast

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('dataset/starcoderdata/self-oss-instruct-sc2-exec-filter-50k.jsonl.code.with.inst') as f:
	responses = [redecode(eval(line)['response'].strip()) for line in f.readlines()]
with open('dataset/starcoderdata/self-oss-instruct-sc2-exec-filter-50k.jsonl.code.with.inst-summarized-by-starcoder2-instruct-reproduce-humaneval') as f:
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

with jsonlines.open('dataset/starcoderdata/self-oss-instruct-sc2-exec-filter-50k.jsonl.code.with.inst-instructed-by-starcoder2-instruct-reproduce-humaneval', mode='w') as writer:
    for i, inst in enumerate(instructions):
        j = i // num_samples
        line = {
	    'instruction' : inst,
	    'response' : responses[j]
	}
        writer.write(line)
