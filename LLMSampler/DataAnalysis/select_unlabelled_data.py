import json
import os
import tqdm
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open('magicoder_data/train-00000-of-00059.json') as f:
    dataset = json.loads(f.read())

model_path = "wizardcoder_gpt4"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_token_length(tokenizer, sequence):
    return tokenizer(sequence, return_tensors='pt')['input_ids'].shape[1]

token_length = []
with tqdm.tqdm(total=len(dataset)) as pbar:
    for i, x in enumerate(dataset):
        code = x['content']
        token_length.append(get_token_length(tokenizer, code))
        pbar.update(1)
print(max(token_length), min(token_length), sum(token_length) / len(token_length))
