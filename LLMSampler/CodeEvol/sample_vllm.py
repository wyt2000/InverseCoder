from vllm import LLM, SamplingParams
import fire
import jsonlines

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
{response}"""

def generate_one_prompt(code):
    # Fill prompt template with one code snippet.
    instruction = f'''Please give a completely new code snippet in the same domain of the following code snippet with more complex and different application:
{code}'''
    prompt =  MAGICODER_PROMPT.format(instruction=instruction, response="```\n")
    prompt =  '[INST]' + prompt + '[/INST]'
    return prompt

def generate_prompts(input_path):
    prompts = []
    with open(input_path, 'r') as f:
        for line in f.readlines():
            line = eval(line)
            code = line['response']
            prompts.append(generate_one_prompt(code))
    return prompts

def extract_code(content: str):
    code = []
    is_target = False
    for line in content.splitlines():
        if is_target:
            code.append(line)
        if '`' in line:
            if is_target:
                break
            else:
                is_target = True
                code.append(line)
    if not code:
        return content
    return '\n'.join(code)

def sample(llm, sampling_params, prompts, save_path):
    # Generate response in parallel and save in the target file.
    outputs = llm.generate(prompts, sampling_params)
    with jsonlines.open(save_path, mode='a') as writer:
        for x in outputs:
            prompt = x.prompt.encode('utf-8', 'backslashreplace').decode('utf-8')
            response = x.outputs[0].text
            # response = extract_code(response)
            response = response.encode('utf-8', 'backslashreplace').decode('utf-8')
            # print(prompt)
            # print(response)
            data = {'instruction': prompt, 'response': response}
            writer.write(data)

def main(
    model_path: str = "model_path",
    num_samples: int = 1,
    input_path: str = "codes.jsonl",
    save_path: str = "samples.jsonl",
    temperature: int = 0.8,
    presence_penalty: float = 0.0,
    frequency_penalty: float = 0.0,
    repetition_penalty: float = 1.1,
    use_beam_search: bool = False,
    best_of: int = 1,
    max_tokens: int = 2048,
    batch_size: int = 512,
    num_gpus: int = 8,
):
    llm = LLM(model=model_path, tensor_parallel_size=num_gpus)
    sampling_params = SamplingParams(
        n=num_samples,
        temperature=temperature,
        max_tokens=max_tokens,
        presence_penalty=presence_penalty,
        frequency_penalty=frequency_penalty,
        repetition_penalty=repetition_penalty,
        use_beam_search=use_beam_search,
        best_of=best_of
    )
    with open(input_path, 'r') as f:
        prompts = []
        for line in f.readlines():
            line = eval(line)
            code = line['response']
            # code = extract_code(code)
            prompts.append(generate_one_prompt(code))
            if len(prompts) == batch_size:
                sample(llm, sampling_params, prompts, save_path)
                prompts = []
        if prompts:
            sample(llm, sampling_params, prompts, save_path)

if __name__ == '__main__':
    fire.Fire(main)
