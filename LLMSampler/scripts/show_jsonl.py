with open('magicoder_data/oss-instruct-codevol-0306-problem-prompt-instruction-data-cleaned-removed.jsonl') as f:
    for line in list(f.readlines())[:1]:
        line = eval(line)
        print('@@instr')
        print(line['instruction'])
        print('@@resp')
        print(line['response'])
