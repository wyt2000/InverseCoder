with open('samples.jsonl') as f:
	for line in list(f.readlines()):
		line = eval(line)
		print(line['instruction'])
		print(line['response'])
