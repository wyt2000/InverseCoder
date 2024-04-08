import jsonlines

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

dataset = []
with open('magicoder_data/evol-instruct-gpt4-data.jsonl') as f:
    lines = list(f.readlines())
    for line in lines:
        line = eval(line)
        instruction = eval(line['query'])[1]['content']
        response = line['response']
        dataset.append({'instruction': instruction, 'response': response})

with jsonlines.open('magicoder_data/evol-instruct-gpt4-summarized-by-gpt35-0407.jsonl', mode='w') as writer:
    for data in dataset:
        writer.write(data)
    
