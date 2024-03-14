import json
import os
import tqdm
import jsonlines
import re
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "false"

with open('magicoder_data/starcoderdata.json') as f:
    dataset = json.loads(f.read())

model_path = "wizardcoder_gpt4_40g"
tokenizer = AutoTokenizer.from_pretrained(model_path)

def get_token_length(tokenizer, sequence):
    return tokenizer(sequence, return_tensors='pt')['input_ids'].shape[1]

max_token_length = 800
min_token_length = 50
save_path = 'magicoder_data/starcoderdata_cleaned.jsonl'
with jsonlines.open(save_path, mode='w') as writer:
    with tqdm.tqdm(total=len(dataset)) as pbar:
        for i, x in enumerate(dataset):
            code = x['content']
            if code.startswith('<gh_stars>') or code.startswith('<reponame>') or code.startswith('<filename>'):
                code = '\n'.join(code.splitlines()[1:])
            token_length = get_token_length(tokenizer, code)
            if min_token_length <= token_length <= max_token_length:
                data = {'raw_code': code.encode('utf-8', 'backslashreplace').decode('utf-8')}
                writer.write(data)
            pbar.update(1)
