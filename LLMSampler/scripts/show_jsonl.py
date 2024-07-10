import json
input_path = '/lustre/S/wuyt/Repoformer/repo_eval/processed_data/python_function_completion.jsonl'

with open(input_path) as f:
    for line in list(f.readlines())[:1]:
        line = json.loads(line)
        for key, value in line.items():
            print(f'@@{key}')
            print(value)
