import os
import sys
from multiprocessing import Pool, current_process, Lock
import argparse
from typing import List
from functools import partial

import fire
import jsonlines
import time
import random

random.seed(42)

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable instructions to user responses.

@@ Instruction
{instruction}

@@ Response
{response}"""

global_prefix = '```python\n'

def generate_one_prompt(inst):
    prompt = MAGICODER_PROMPT.format(instruction=inst, response=global_prefix)
    return prompt

def generate_prompts(input_path):
    prompts = []
    with open(input_path, 'r') as f:
        for line in f.readlines():
            line = eval(line)
            code = line['response']
            prompts.append(generate_one_prompt(code))
    return prompts

def extract_code(code: str):
    if not '```' in code:
        return code
    start = code.find('```')
    end = code.rfind('```')
    ret = code[start:end]
    # ret = '\n'.join(ret.splitlines()[1:])
    if not ret:
        ret = code[start:]
        # ret = '\n'.join(ret.splitlines()[1:])
    return ret

def get_language(response):
    for line in response.splitlines():
        if '```' in line:
            return line.split('```')[1]
    return ''

    
def sample(llm, sampling_params, prompts, save_path):
    # Generate response in parallel and save in the target file.

    outputs = llm.generate(prompts, sampling_params, use_tqdm=False)
    results = []
    with jsonlines.open(save_path, mode='a') as writer:
        for x in outputs:
            prompt = x.prompt.encode('utf-8', 'backslashreplace').decode('utf-8')
            for output in x.outputs:
                response = output.text
                # response = extract_code(response)
                response = response.encode('utf-8', 'backslashreplace').decode('utf-8')
                # print(prompt)
                # print(response)
                data = {'instruction': prompt, 'response': response}
                writer.write(data)
                results.append(data)
    return results

def main(
    input_lines: List[str],
    model_path: str,
    save_path: str,
    deepseek: bool,
    num_samples: int = 1,
    temperature: int = 0.0,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.0,
    use_beam_search: bool = False,
#    best_of: int = 1,
    max_tokens: int = 2048,
    stop: List[str] = ['```'], 
#    stop: List[str] = [' Here', '\nHere'], 
    batch_size: int = 4 
):
    pid = int(current_process()._identity[0]) - 1
    print(f'[Parallel] pid: {pid}, data size: {len(input_lines)}')
    os.environ["CUDA_VISIBLE_DEVICES"] = str(pid)
    save_path = f'{save_path}.{pid}'

    from vllm import LLM, SamplingParams
    with lock:
        llm = LLM(model=model_path, max_model_len=32800) if deepseek else LLM(model=model_path)
    sampling_params = SamplingParams(
        n=1,
        temperature=temperature,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        use_beam_search=use_beam_search,
#        best_of=best_of,
        stop=stop
    )
    
    def generate_with_timer(prompts):
        start_time = time.perf_counter()
        results = sample(llm, sampling_params, prompts, save_path)
        end_time = time.perf_counter()
        print(f'[Parallel] pid: {pid}, generated data: {len(prompts)}, time: {end_time - start_time}s')
        return results

    input_dataset = []
    for line in input_lines:
        for i in range(num_samples):
            input_dataset.append(line)

    prompts = []
    results = []
    for line in input_dataset:
        line = eval(line)
        code = line['prompt']
        # code = extract_code(code) + '```'
        prompts.append(generate_one_prompt(code))
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
    parser.add_argument('--deepseek', action='store_true',)
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
        results = p.map(partial(main, model_path=args.model_path, save_path=args.save_path, deepseek=args.deepseek), data_chunks)
        
    i = 0
    with jsonlines.open(args.save_path, mode='w') as writer:
        for result in results:
            for data in result:
                line = eval(dataset[i])
                i += 1
                x = {}
                x['task_id'] = line['task_id']
                x['completion'] = data['response']
                writer.write(x)

