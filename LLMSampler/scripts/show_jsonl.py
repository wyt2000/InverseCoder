with open('magicoder_data/codevol-wizardcoder-gpt4-prompt-modified-1-0312.jsonl.1') as f:
	for line in list(f.readlines())[:10]:
		line = eval(line)
		print(line['instruction'])
		print(line['response'])
