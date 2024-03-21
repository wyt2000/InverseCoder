from datasketch import MinHash, MinHashLSH
import tqdm
import jsonlines

def create_minhash(data):
    minhash = MinHash(num_perm=128)
    for d in data:
        minhash.update(d.encode('utf8'))
    return minhash

benchmark_data = []
with open('magicoder_data/humaneval-benchmark.jsonl') as f:
    for line in f.readlines():
        line = eval(line)
        benchmark_data.append(line['prompt'])

# 创建 MinHash 对象并插入到 LSH 中
lsh = MinHashLSH(threshold=0, num_perm=128)  # threshold 是相似度阈值，可以根据需要调整
for idx, data in enumerate(benchmark_data):
    minhash = create_minhash(list(data))
    lsh.insert(idx, minhash)

dataset = []
with open('magicoder_data/data-evol_instruct-decontaminated.jsonl.python') as f:
    for line in list(f.readlines()):
        line = eval(line)
        dataset.append(line['instruction'])
    
def redecode(s):
    return s.encode('utf-8', 'backslashreplace').decode('utf-8')

with jsonlines.open('magicoder_data/data-evol_instruct-decontaminated.jsonl.python.minhash', mode='w') as writer:
    with tqdm.tqdm(total=len(dataset)) as pbar:
        for data in dataset:
            # 查找相似的集合
            query_minhash = create_minhash(list(data))
            results = lsh.query(query_minhash)
            # 输出相似度分数
            similar_data = ('', -1)
            for result in results:
                minhash = create_minhash(list(benchmark_data[result]))
                jaccard_similarity = query_minhash.jaccard(minhash)
                if jaccard_similarity > similar_data[1]:
                    similar_data = (benchmark_data[result], jaccard_similarity)
            writer.write({'instruction': redecode(data), 'closest_test': similar_data[0], 'similarity': similar_data[1]})
            pbar.update(1)