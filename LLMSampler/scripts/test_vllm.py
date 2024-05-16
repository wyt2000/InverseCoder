from vllm import LLM, SamplingParams

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
{response}"""

import json
input_path = '/lustre/S/wuyt/dataset/evol-instruct-gpt4/data-evol_instruct-decontaminated.jsonl.code'

inst = '''Please rewrite the code in another way while keeping the functionality unchanged:
{code}'''

def get_language(response):
    for line in response.splitlines():
        if '```' in line:
            return line.split('```')[1]
    return ''

prompts = []
with open(input_path) as f:
    for line in list(f.readlines())[:5]:
        line = json.loads(line)
        lang = get_language(line['response'])
        prompts.append(MAGICODER_PROMPT.format(instruction=inst.format(code=line['response']), response='```'+lang))

model_path = '../model/wizardcoder-gpt4-reproduce-0424'
sampling_params = SamplingParams(temperature=0, max_tokens=2048)
llm = LLM(model=model_path)

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
