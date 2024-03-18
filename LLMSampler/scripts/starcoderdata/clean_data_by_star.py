import jsonlines
import tqdm
from collections import Counter

num_data = 300000

repo_cnt = Counter()
dataset = []
with tqdm.tqdm(total=num_data) as pbar:
    with open('starcoderdata-python-jsonl/starcoderdata-python-sorted.jsonl') as f:
        while True:
            if len(dataset) == num_data: break
            line = f.readline()
            if not line: break
            line = eval(line)
            repo_name = line['max_stars_repo_name'].split('/')[-1]
            stars = line['max_stars_count']
            if repo_cnt[(repo_name, stars)] >= 3: continue
            dataset.append(line)
            repo_cnt[(repo_name, stars)] += 1
            pbar.update(1)

with tqdm.tqdm(total=num_data) as pbar:
    with jsonlines.open('starcoderdata-python-jsonl/starcoderdata-python-top300k.jsonl', mode='w') as writer:
        for data in dataset:
            writer.write(data)
            pbar.update(1)