# input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl.fixed.python.instruct-0324'
# input_path = 'magicoder_data/data-evol_instruct-decontaminated.jsonl.no_python.summarized.by.deepseek.0412.0'
# input_path = '/lustre/S/wuyt/dataset/starcoderdata/self-oss-instruct-sc2-exec-filter-50k.jsonl-instructed-by-wizardcoder-gpt4-reproduce-0424-problem-prompt'
input_path = 'dataset/oss-instruct/oss-instruct.jsonl.code-instructed-by-magicoder-DS-reproduce-0501-select-by-magicoder-DS-reproduce-0501-with-score-sorted-instruction-completed-by-magicoder-DS-reproduce-0501.0'
import json
# input_path = 'dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code-instructed-by-wizardcoder-gpt4-reproduce-0424-problem-prompt-samples-10-select-by-wizardcoder-gpt4-reproduce-0424-with-score-sorted'

with open(input_path) as f:
    for line in list(f.readlines())[:10]:
        line = json.loads(line)
        print('@@instruction')
        print(line['instruction'])
        print('@@response')
        print(line['response'])
