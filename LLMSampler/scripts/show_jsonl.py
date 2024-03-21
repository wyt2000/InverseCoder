with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.python') as f:
    for line in list(f.readlines())[10:20]:
        line = eval(line)
        print('@@instr')
        print(line['instruction'])
        print('@@resp')
        print(line['response'])
