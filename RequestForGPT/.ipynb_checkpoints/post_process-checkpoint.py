import jsonlines

def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

dataset = []
with open('magicoder_data/evol-instruct-gpt4-summarized-by-gpt35-0407.jsonl') as f:
    lines = list(f.readlines())
    for line in lines:
        line = eval(line)
        response = line['instruction']
        response = response.split('The Code Snippet:\n')[1]
        instruction = line['response']
        instruction = instruction[
            : index
            if (index := instruction.find("```")) != -1
            else len(instruction)
        ]
        dataset.append({'instruction': instruction, 'response': response})

with jsonlines.open('magicoder_data/evol-instruct-gpt4-summarized-by-gpt35-0407-processed.jsonl', mode='w') as writer:
    for data in dataset:
        writer.write(data)
