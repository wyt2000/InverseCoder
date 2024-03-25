# input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.python.instruct-0324'
input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.no_python.evol-0325.0'

with open(input_path) as f:
    for line in list(f.readlines()):
        line = eval(line)
        if 'prime' not in line['instruction']: continue
        print('@@instr')
        print(line['instruction'])
        print('@@resp')
        print(line['response'])
