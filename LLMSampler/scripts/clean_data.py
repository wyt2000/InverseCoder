# Remove bad data
import jsonlines

with open('magicoder_data/oss-instruct-codevol-0306-no-prompt-instruction-data-cleaned.jsonl') as f:
    no_prompt_lines = list(f.readlines())

with open('magicoder_data/oss-instruct-codevol-0306-problem-prompt-instruction-data-cleaned.jsonl') as f:
    problem_prompt_lines = list(f.readlines())


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

with jsonlines.open('magicoder_data/oss-instruct-codevol-0306-problem-prompt-instruction-data-cleaned-selected.jsonl', mode='w') as writer:
    removed_data = []
    for no_prompt_line, problem_prompt_line in zip(no_prompt_lines, problem_prompt_lines):
        no_prompt_line = eval(no_prompt_line)
        no_prompt_line['instruction'] = redecode(no_prompt_line['instruction'])
        no_prompt_line['response'] = redecode(no_prompt_line['response'])
        problem_prompt_line = eval(problem_prompt_line)
        problem_prompt_line['instruction'] = redecode(problem_prompt_line['instruction'])
        problem_prompt_line['response'] = redecode(problem_prompt_line['response'])
        if check(no_prompt_line['instruction'], no_prompt_line['response']):
            writer.write(problem_prompt_line)
        else:
            removed_data.append(no_prompt_line)

with jsonlines.open('magicoder_data/oss-instruct-codevol-0306-problem-prompt-instruction-data-cleaned-removed.jsonl', mode='w') as writer:
    for line in removed_data:
        writer.write(line)

print(f'Remove {len(removed_data)} data!')

