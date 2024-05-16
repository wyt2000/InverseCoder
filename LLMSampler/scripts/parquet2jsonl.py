import pyarrow.parquet as pq
import jsonlines
import json

def read_parquet(path):
    df = pq.read_table(path)
    df = df.to_pandas()
    dataset = df.to_json(orient='records')
    dataset = json.loads(dataset)
    return dataset

raw_dataset = read_parquet('/lustre/S/wuyt/dataset/starcoderdata/self-oss-instruct-sc2-exec-filter-50k.parquet')
dataset = []

for data in raw_dataset:
    dataset.append(data)
    # dataset.append({'response' : data['content']})

with jsonlines.open(f'/lustre/S/wuyt/dataset/starcoderdata/self-oss-instruct-sc2-exec-filter-50k.jsonl', mode='w') as writer:
    for data in dataset:
        writer.write(data)
