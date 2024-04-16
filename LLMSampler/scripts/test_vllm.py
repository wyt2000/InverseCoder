from vllm import LLM, SamplingParams

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
{response}"""

inst = '''Write a function to find the nth tetrahedral number. A tetrahedral number, or triangular pyramidal number, is a figurate number that represents a pyramid with a triangular base and three sides, called a tetrahedron. The nth tetrahedral number, Tn, is the sum of the first n triangular numbers, that is:
Tn = n * (n + 1) * (n + 2) / 6
Your code should satisfy the following assertion:
```python
assert tetrahedral_number(5) == 35
```'''

prompts = [
    MAGICODER_PROMPT.format(instruction=inst, response='```python\ndef tetrahedral_number(n):'),
]
model_path = 'wizardcoder-gpt4-deduplicated-with-no-python-summary-only-by-python-model-0329/'
sampling_params = SamplingParams(temperature=0, max_tokens=2048)
llm = LLM(model=model_path)

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")