# input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.python.instruct-0324'
# input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl.no_python.summarized.by.deepseek.0412.0'
input_path = 'dataset/data-evol_instruct-decontaminated.jsonl.no_python.evol-0327.instruct.all_print'

with open(input_path) as f:
    for line in list(f.readlines())[:10]:
        line = eval(line)
        print('@@instr')
        print(line['instruction'])
        print('@@resp')
        print(line['response'])
