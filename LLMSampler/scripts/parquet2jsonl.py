import pyarrow.parquet as pq
import jsonl

df = pq.read_table('train-00000-of-00059.parquet')
df = df.to_pandas()
json_data = df.to_json(orient='records')
with open('train-00000-of-00059.json', 'w') as f:
    f.write(json_data)