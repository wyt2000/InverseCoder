import random
import pandas as pd
import jsonlines
random.seed(42)

with open('magicoder_data/codevol-with-summary-0220.jsonl') as f:
	lines = list(f.readlines())
	sample_lines = random.sample(lines, 20)
	with jsonlines.open('magicoder_data/codevol-with-summary-20r.jsonl', mode='w') as writer:
		for line in sample_lines:
			line = eval(line)
			writer.write(line)