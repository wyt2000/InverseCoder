import jsonlines

data_prefix = 'dataset/data-evol_instruct-decontaminated'

with open(f'{data_prefix}.jsonl') as f:
    data_lines = list(f.readlines())

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
with jsonlines.open(f'{data_prefix}_cleaned.jsonl', mode='w') as writer:
    for data in data_lines:
        data = eval(data)
        data['instruction'] = redecode(data['instruction'])
        data['response'] = redecode(data['response'])
        if check(data['instruction'], data['response']):
            writer.write(data)
        else:
            removed_data.append(data)

with jsonlines.open(f'{data_prefix}_removed.jsonl', mode='w') as writer:
    for line in removed_data:
        writer.write(line)

print(f'Total {len(removed_data)} data!')

