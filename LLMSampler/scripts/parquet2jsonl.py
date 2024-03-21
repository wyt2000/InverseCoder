import pyarrow.parquet as pq
import jsonlines
import json

# for i in range(59):
#    i = str(i).rjust(2, '0')
with jsonlines.open(f'magicoder_data/humaneval-benchmark.jsonl', mode='w') as writer:
    df = pq.read_table(f'magicoder_data/humaneval.parquet')
    df = df.to_pandas()
    dataset = df.to_json(orient='records')
    dataset = json.loads(dataset)
    for data in dataset:
        writer.write(data)
