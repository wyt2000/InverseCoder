import json
import tqdm
import jsonlines
import matplotlib.pyplot as plt

stars = []

for i in range(59):
    i = str(i).rjust(2, '0')
    with open(f'starcoderdata-python-jsonl/starcoderdata-python-jsonl-{i}.jsonl', 'r') as f:
        dataset = f.readlines()
        with tqdm.tqdm(total=len(dataset)) as pbar:
            for data in dataset:
                data = eval(data)
                stars.append(data['max_stars_count'])
                pbar.update(1)

plt.figure()
plt.hist(stars, 10, (1, 10)) 
plt.grid(alpha=0.5,linestyle='-.')
plt.savefig('starcoderdata.png')
