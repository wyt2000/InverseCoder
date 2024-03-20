import jsonlines
import tqdm

min_token_length = 0
max_token_length = 1024

dataset = []

cnt = 0
with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.token_count') as f:
    while True:
        line = f.readline()
        if not line: break
        line = eval(line)
        if min_token_length <= line['token_count'] <= max_token_length:
            dataset.append(line)
        else:
            cnt += 1

with tqdm.tqdm(total=len(dataset)) as pbar:
    with jsonlines.open('magicoder_data/data-evol_instruct-decontaminated-truncated-1024.jsonl', mode='w') as writer:
        for data in dataset:
            writer.write(data)
            pbar.update(1)
print(f'Remove {cnt} data!')
