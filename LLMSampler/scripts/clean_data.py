# Remove bad data
import jsonlines

with open('magicoder_data/codevol-wizardcoder-0304-instruct-by-wizardcoder-reversed-0306-0311-cleaned.jsonl') as f:
    lines = list(f.readlines())

def check(inst, resp):
    s = ''.join([inst, resp])
    # if 'OCR' in inst or 'apologize' in s or 'apologies' in s or 'apology' in s:
    if 'apologize' in s or 'apologies' in s or 'apology' in s:
        return False
    return True

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with jsonlines.open('magicoder_data/codevol-wizardcoder-0304-instruct-by-wizardcoder-reversed-0306-0311-cleaned-selected.jsonl', mode='w') as writer:
    removed_data = []
    for line in lines:
        line = eval(line)
        line['instruction'] = redecode(line['instruction'])
        line['response'] = redecode(line['response'])
        if check(line['instruction'], line['response']):
            writer.write(line)
        else:
            removed_data.append(line)

with jsonlines.open('magicoder_data/codevol-wizardcoder-0304-instruct-by-wizardcoder-reversed-0306-0311-removed.jsonl', mode='w') as writer:
    for line in removed_data:
        writer.write(line)

print(f'Remove {len(removed_data)} data!')

