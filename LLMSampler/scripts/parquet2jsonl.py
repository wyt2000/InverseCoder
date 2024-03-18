import pyarrow.parquet as pq
import jsonlines
import json

for i in range(59):
    i = str(i).rjust(2, '0')
    with jsonlines.open(f'starcoderdata-python-jsonl/starcoderdata-python-jsonl-{i}.jsonl', mode='w') as writer:
        df = pq.read_table(f'starcoderdata-python/train-000{i}-of-00059.parquet')
        df = df.to_pandas()
        dataset = df.to_json(orient='records')
        dataset = json.loads(dataset)
        for data in dataset:
            writer.write(data)
