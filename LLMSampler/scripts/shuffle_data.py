import random
import pandas as pd
import jsonlines
random.seed(42)

with open('magicoder_data/data-evol_instruct-decontaminated.jsonl') as f:
	lines = list(f.readlines())
    lines = random.shuffle(lines)
	with jsonlines.open('magicoder_data/data-evol_instruct-decontaminated-shuffled.jsonl', mode='w') as writer:
		for line in lines:
			line = eval(line)
			writer.write(line)