import random
import pandas as pd
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.no_python') as f:
	lines = list(f.readlines())
	random.shuffle(lines)
	with jsonlines.open('magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.all_with_instructed_python_0324.jsonl', mode='w') as writer:
		for line in lines:
			line = eval(line)
			line['instruction'] = redecode(line['instruction'])
			line['response'] = redecode(line['response'])
			writer.write(line)