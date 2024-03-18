with open('magicoder_data/starcoderdata-python-topstar-evol.jsonl.1') as f:
	for line in list(f.readlines())[:10]:
		line = eval(line)
		print(line['instruction'])
		print(line['response'])
