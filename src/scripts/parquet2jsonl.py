import pyarrow.parquet as pq
import jsonlines
import json

def read_parquet(path):
    df = pq.read_table(path)
    df = df.to_pandas()
    dataset = df.to_json(orient='records')
    dataset = json.loads(dataset)
    return dataset

raw_dataset = read_parquet('dataset/starcoderdata/python-stack-v1-functions-filtered-sc2.parquet')
dataset = []

for data in raw_dataset:
    data['seed'] = data['content']
    del data['content']
    dataset.append(data)
    # dataset.append({'response' : data['content']})

with jsonlines.open(f'dataset/starcoderdata/python-stack-v1-functions-filtered-sc2.jsonl', mode='w') as writer:
    for data in dataset:
        writer.write(data)
