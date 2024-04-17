import random
import pandas as pd
import jsonlines
random.seed(42)

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with open('dataset/data-evol_instruct-decontaminated.jsonl.no_python-instruct-by-deepseekcoder-6.7b-instruct.with.python.data') as f:
	lines = list(f.readlines())
	random.shuffle(lines)
	with jsonlines.open('dataset/data-evol_instruct-decontaminated.jsonl.no_python-instruct-by-deepseekcoder-6.7b-instruct.with.python.data.shuffled', mode='w') as writer:
		for line in lines:
			line = eval(line)
			line['instruction'] = redecode(line['instruction'])
			line['response'] = redecode(line['response'])
			writer.write(line)
