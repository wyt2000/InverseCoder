from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import numpy as np
import json
import os
import random
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer, AutoModelForCausalLM
import jsonlines


model_path = '../model/wizardcoder-gpt4'

def get_token_ids(tokens):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    token_ids = {
        token : tokenizer(token, return_tensors='pt')['input_ids'][0][1].item()
        for token in tokens
    }
    return token_ids

token_ids = get_token_ids(['YES', 'Yes', 'NO', 'No'])

def get_tokens_logprob(probs, tokens):
    ans = -100
    for token in tokens:
        ans = max(ans, probs.get(token_ids[token], -100))
    return ans

def get_yes_prob(probs):
    print(probs)
    yes_logprob = get_tokens_logprob(probs, ['YES','Yes'])
    no_logprob = get_tokens_logprob(probs, ['NO', 'No'])
    print(yes_logprob)
    print(no_logprob)
    return np.exp(yes_logprob) / (np.exp(yes_logprob) + np.exp(no_logprob))

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
{response}"""

inst = '''Code Snippet:
```python
{code}
```
Does the code contain elements of mathematical intelligence? Reply with only YES or NO.'''

prompts = [
    MAGICODER_PROMPT.format(instruction=inst.format(code=code1), response=''),
    MAGICODER_PROMPT.format(instruction=inst.format(code=code2), response=''),
    MAGICODER_PROMPT.format(instruction=inst.format(code=code3), response=''),
]
sampling_params = SamplingParams(temperature=0, max_tokens=2048, logprobs=5)
llm = LLM(model=model_path)

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
    yes_prob = get_yes_prob(output.outputs[0].logprobs[0])
    print(f'YES prob: {yes_prob}')

