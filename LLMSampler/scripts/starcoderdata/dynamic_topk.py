import json
import tqdm
import jsonlines
import matplotlib.pyplot as plt

data_with_stars = []

for i in range(59):
    i = str(i).rjust(2, '0')
    with open(f'starcoderdata-python-jsonl/starcoderdata-python-jsonl-{i}.jsonl', 'r') as f:
        dataset = f.readlines()
        with tqdm.tqdm(total=len(dataset)) as pbar:
            for data in dataset:
                data = eval(data)
                data_with_stars.append(data)
                pbar.update(1)

data_with_stars = sorted(data_with_stars, key=lambda x : x['max_stars_count'])