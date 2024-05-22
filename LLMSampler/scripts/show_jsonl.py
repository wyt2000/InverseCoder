import json
input_path = 'dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl'

with open(input_path) as f:
    for line in list(f.readlines())[:1]:
        line = json.loads(line)
        print('@@instruction')
        print(line['instruction'])
        print('@@response')
        print(line['response'])
