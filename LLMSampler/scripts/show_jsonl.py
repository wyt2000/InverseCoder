with open('magicoder_data/starcoderdata_cleaned_0314_instructed_by_comment_0319.jsonl') as f:
    for line in list(f.readlines())[-20:-10]:
        line = eval(line)
        print('@@instr')
        print(line['instruction'])
        print('@@resp')
        print(line['response'])
