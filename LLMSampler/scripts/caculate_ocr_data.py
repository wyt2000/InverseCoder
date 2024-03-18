import jsonlines

with open('magicoder_data/data-evol_instruct-decontaminated.jsonl') as f:
    no_prompt_lines = list(f.readlines())

def check(inst, resp):
    s = ''.join([inst, resp])
    if 'OCR' in inst:
        return False
    # if 'apologize' in s or 'apologies' in s or 'apology' in s:
    #    return False
    #if '```' in inst:
    #    return False
    return True

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')


removed_data = []
for no_prompt_line in no_prompt_lines:
    no_prompt_line = eval(no_prompt_line)
    no_prompt_line['instruction'] = redecode(no_prompt_line['instruction'])
    no_prompt_line['response'] = redecode(no_prompt_line['response'])
    if check(no_prompt_line['instruction'], no_prompt_line['response']):
        pass
    else:
        removed_data.append(no_prompt_line)

print(f'Total {len(removed_data)} data!')

