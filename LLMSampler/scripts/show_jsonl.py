with open('magicoder_data/oss-instruct-codevol-0306-summary-3200.jsonl') as f:
	for line in list(f.readlines())[:10]:
		line = eval(line)
		print(line['instruction'])
		print(line['response'])
