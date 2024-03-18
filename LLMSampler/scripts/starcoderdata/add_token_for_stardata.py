import jsonlines
import tqdm
from collections import Counter

import json
import os
from transformers import AutoTokenizer
from multiprocessing import Pool, current_process, Lock

os.environ["TOKENIZERS_PARALLELISM"] = "false"
model_path = "wizardcoder_gpt4_40g"
input_path = 'starcoderdata-python-jsonl/starcoderdata-python-top300k.jsonl'
save_path = 'starcoderdata-python-jsonl/starcoderdata-python-top300k-with-token-count.jsonl'

max_token_length = 800
min_token_length = 50
# num_data = 112633
num_data = 300000

def add_token(data_lines):
    batch_size = 512
    pid = int(current_process()._identity[0]) - 1
    print(f'[Parallel] pid: {pid}, data size: {len(data_lines)}')
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    def get_token_length(tokenizer, sequence):
        return tokenizer(sequence, return_tensors='pt')['input_ids'].shape[1]
    results = []
    for i, line in enumerate(data_lines):
        line = eval(line)
        code = line['content']
        if code.startswith('<gh_stars>') or code.startswith('<reponame>') or code.startswith('<filename>'):
            code = '\n'.join(code.splitlines()[1:])
        token_length = get_token_length(tokenizer, code)
        line['content'] = code
        line['token_length'] = token_length
        results.append(line)
        if (i + 1) % batch_size == 0:
            print(f'[Parallel] pid: {pid}, generated data: {batch_size}')
    return results

if __name__ == '__main__':

    dataset = []
    with open(input_path, 'r') as f:
        for data in f.readlines():
            dataset.append(data)

    num_processes = 64
    num_data_per_process = (len(dataset) + num_processes - 1) // num_processes
    data_chunks = [[] for i in range(num_processes)]
    for i, data in enumerate(dataset):
        data_chunks[i // num_data_per_process].append(data)

    with Pool(num_processes) as p:
        results = p.map(add_token, data_chunks)
        
    with jsonlines.open(save_path, mode='w') as writer:
        for result in results:
            for data in result:
                writer.write(data)