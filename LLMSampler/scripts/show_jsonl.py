# input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.python.instruct-0324'
# input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl.no_python.summarized.by.deepseek.0412.0'
input_path = 'magicoder_data/AutoMathTextData/0.75-1.00.jsonl.summarized.0413.0'

with open(input_path) as f:
    for line in list(f.readlines())[30:40]:
        line = eval(line)
        print('@@instr')
        print(line['instruction'])
        print('@@resp')
        print(line['response'])
