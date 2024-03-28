# input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.python.instruct-0324'
input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct'

with open(input_path) as f:
    for line in list(f.readlines())[0:10]:
        line = eval(line)
        print('@@instr')
        print(line['instruction'])
        print('@@resp')
        print(line['response'])
