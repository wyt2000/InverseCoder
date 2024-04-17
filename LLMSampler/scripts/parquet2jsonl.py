import pyarrow.parquet as pq
import jsonlines
import json

def read_parquet(path):
    df = pq.read_table(path)
    df = df.to_pandas()
    dataset = df.to_json(orient='records')
    dataset = json.loads(dataset)
    return dataset

dataset = []
dataset.extend(read_parquet('magicoder_data/mbpp_data/prompt-00000-of-00001.parquet'))
dataset.extend(read_parquet('magicoder_data/mbpp_data/test-00000-of-00001.parquet'))
dataset.extend(read_parquet('magicoder_data/mbpp_data/train-00000-of-00001.parquet'))
dataset.extend(read_parquet('magicoder_data/mbpp_data/validation-00000-of-00001.parquet'))

with jsonlines.open(f'magicoder_data/mbpp-benchmark.jsonl', mode='w') as writer:
    for data in dataset:
        writer.write(data)
