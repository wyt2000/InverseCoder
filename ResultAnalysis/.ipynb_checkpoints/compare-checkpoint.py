import json
import jsonlines

ref_result_path = 'magicoder_data/mbpp-0403_eval_results.json'
ours_result_path = 'magicoder_data/mbpp-0407_eval_results.json'

ref_generation_path = 'magicoder_data/mbpp-0403.jsonl'
ours_generation_path = 'magicoder_data/mbpp-0407.jsonl'

def get_results(path):
    with open(path, 'r') as f:
        data = json.load(f)
    status = []
    for name, value in data['eval'].items():
        status.append((name, value['base'][0][0]))
    return sorted(status)

def get_generations(path):
    generations = {}
    with open(path, 'r') as f:
        for line in f.readlines():
            line = eval(line)
            generations[line['task_id']] = line['completion']
    return generations

ref_results = get_results(ref_result_path)
ref_generations = get_generations(ref_generation_path)
ours_results = get_results(ours_result_path)
ours_generations = get_generations(ours_generation_path)

for x, y in zip(ref_results, ours_results):
    if x != y:
        print(f'ref: {x}, ours: {y}')
        print(ref_generations[x[0]])
        print(ours_generations[y[0]])