import os
import sys
from multiprocessing import Pool, current_process, Lock
import argparse
from typing import List
from functools import partial

import fire
import jsonlines
import time

from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
{response}"""

def get_token_ids(model_path, tokens):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_ids = {
        token : tokenizer(token, return_tensors='pt')['input_ids'][0][1].item()
        for token in tokens
    }
    return token_ids

def get_tokens_logprob(probs, tokens, token_ids):
    ans = -100
    for token in tokens:
        ans = probs.get(token_ids[token], -100)
        if ans != -100:
            ans = ans.logprob
            break
    return ans

def get_yes_prob(probs, yes_tokens, no_tokens, token_ids):
    yes_logprob = get_tokens_logprob(probs, yes_tokens, token_ids)
    no_logprob = get_tokens_logprob(probs, no_tokens, token_ids)
    return np.exp(yes_logprob) / (np.exp(yes_logprob) + np.exp(no_logprob))

def generate_one_prompt(inst, code):
    # Fill prompt template with one code snippet.
    instruction = f'''Here is a programming problem:
{inst}

Here is the answer code to the problem:
{code}
Is the answer correct? Your reply should begin with Yes or No.'''
    prompt =  MAGICODER_PROMPT.format(instruction=instruction, response="")
    return prompt

def generate_prompts(input_path):
    prompts = []
    with open(input_path, 'r') as f:
        for line in f.readlines():
            line = eval(line)
            code = line['raw_code']
            prompts.append(generate_one_prompt(code))
    return prompts

def extract_code(code: str):
    if not '```' in code:
        return code
    start = code.find('```')
    end = code.rfind('```')
    ret = code[start:end]
    ret = '\n'.join(ret.splitlines()[1:])
    if not ret:
        ret = code[start:]
        ret = '\n'.join(ret.splitlines()[1:])
    return ret

def sample(llm, token_ids, sampling_params, prompts, save_path):
    # Generate response in parallel and save in the target file.
    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    results = []
    with jsonlines.open(save_path, mode='a') as writer:
        for x in outputs:
            prompt = x.prompt.encode('utf-8', 'backslashreplace').decode('utf-8')
            response = x.outputs[0].text
            yes_prob = x.outputs[0].logprobs[0]
            yes_prob = get_yes_prob(yes_prob, ['Yes'], ['No', 'As'], token_ids)
            data = {'instruction': prompt, 'response': response, 'yes_prob': yes_prob}
            writer.write(data)
            results.append(data)
    return results

def main(
    input_lines: List[str],
    model_path: str,
    save_path: str,
    num_samples: int = 1,
    temperature: int = 0.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    use_beam_search: bool = False,
    best_of: int = 1,
    max_tokens: int = 5,
    logprobs: int = 5,
    stop: List[str] = [],
    batch_size: int = 512
):
    pid = int(current_process()._identity[0]) - 1
    print(f'[Parallel] pid: {pid}, data size: {len(input_lines)}')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pid) 
    print(f'Using device {os.environ["CUDA_VISIBLE_DEVICES"]}')
    save_path = f'{save_path}.{pid}'
    from vllm import LLM, SamplingParams

    with lock:
        llm = LLM(model=model_path, max_model_len=11792)
        # llm = LLM(model=model_path)
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        use_beam_search=use_beam_search,
        best_of=best_of,
        stop=stop,
        logprobs=logprobs
    )
    
    token_ids = get_token_ids(model_path, ['Yes', 'No', 'As'])

    def generate_with_timer(prompts):
        start_time = time.perf_counter()
        results = sample(llm, token_ids, sampling_params, prompts, save_path)
        end_time = time.perf_counter()
        print(f'[Parallel] pid: {pid}, generated data: {len(prompts)}, time: {end_time - start_time}s')
        return results

    prompts = []
    results = []
    for line in input_lines:
        line = eval(line)
        inst = line['instruction']
        code = line['response']
        # code = extract_code(code)
        prompts.append(generate_one_prompt(inst, code))
        if len(prompts) == batch_size:
            results.extend(generate_with_timer(prompts))
            prompts = []
    if prompts:
        results.extend(generate_with_timer(prompts))
    return results

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--input_path', type=str, required=True)
    parser.add_argument('--save_path', type=str, required=True)
    parser.add_argument('--num_gpus', type=int, required=True)
    args = parser.parse_args()

    with open(args.input_path, 'r') as f:
        dataset = list(f.readlines())

    num_processes = args.num_gpus
    num_data_per_process = (len(dataset) + num_processes - 1) // num_processes
    data_chunks = [[] for i in range(num_processes)]
    for i, data in enumerate(dataset):
        data_chunks[i // num_data_per_process].append(data)

    lock = Lock()
    with Pool(num_processes) as p:
        results = p.map(partial(main, model_path=args.model_path, save_path=args.save_path), data_chunks)
        
    with jsonlines.open(args.save_path, mode='a') as writer:
        for result in results:
            for data in result:
                writer.write(data)

