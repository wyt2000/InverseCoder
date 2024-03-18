import jsonlines
import tqdm

max_token_length = 800
min_token_length = 50
num_data = 112633

dataset = []

with tqdm.tqdm(total=num_data) as pbar:
    with open('starcoderdata-python-jsonl/starcoderdata-python-top300k-with-token-count.jsonl') as f:
        while True:
            if len(dataset) == num_data: break
            line = f.readline()
            if not line: break
            line = eval(line)
            if min_token_length <= line['token_length'] <= max_token_length:
                dataset.append(line)
                pbar.update(1)

with tqdm.tqdm(total=num_data) as pbar:
    with jsonlines.open('starcoderdata-python-jsonl/starcoderdata-python-top300k-selected.jsonl', mode='w') as writer:
        for data in dataset:
            writer.write(data)
            pbar.update(1)
