from vllm import LLM, SamplingParams

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
"""

import json

inst = '''Please check if the network can connect to `www.example.com` using Python.'''

def get_language(response):
    for line in response.splitlines():
        if '```' in line:
            return line.split('```')[1]
    return ''

prompts = []
prompts.append(MAGICODER_PROMPT.format(instruction=MAGICODER_PROMPT.format(instruction=inst)))

model_path = '../model/inversecoder-DS-0510'
sampling_params = SamplingParams(temperature=0, max_tokens=2048)
llm = LLM(model=model_path, max_model_len=32800)

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(prompt)
    print(generated_text)
