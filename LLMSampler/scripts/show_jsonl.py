with open('magicoder_data/starcoderdata_cleaned_0314_with_docstring_0319-comment-prompt.jsonl.0') as f:
    for line in list(f.readlines())[10:20]:
        line = eval(line)
        print('@@instr')
        print(line['instruction'])
        print('@@resp')
        print(line['response'])
