import jsonlines

input_path = 'magicoder_data/AutoMathTextData/0.75-1.00.jsonl.instructed.0413.token_count'
save_path = 'magicoder_data/AutoMathTextData/0.75-1.00.jsonl.instructed.0413.flitered'

with open(input_path, 'r') as f:
    input_lines = list(f.readlines())

max_token_count = 1216

with jsonlines.open(save_path, mode='w') as writer:
    for line in input_lines:
        line = eval(line)
        if not line['instruction'] or line['token_count'] > max_token_count: continue
        writer.write(line)
