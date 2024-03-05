import random
import pandas as pd
import jsonlines
random.seed(42)

with open('magicoder_data/evol-instruct-codevol-0213.jsonl') as f:
	lines = list(f.readlines())
	sample_lines = random.sample(lines, 20)
	with jsonlines.open('magicoder_data/codevol-0213-20r.jsonl', mode='w') as writer:
		for line in sample_lines:
			line = eval(line)
			writer.write(line)
	ans = []
	for line in sample_lines:
		line = eval(line)
		inst = line['instruction']
		res  = line['response']
		df = pd.DataFrame({'instruction': inst, 'response': res}, index=[0])
		ans.append(df)
	df = pd.concat(ans)
	with pd.ExcelWriter('data.xlsx') as writer:
		df.to_excel(writer, sheet_name='data', index=False)
