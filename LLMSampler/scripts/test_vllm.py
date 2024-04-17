from vllm import LLM, SamplingParams

MAGICODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.

@@ Instruction
{instruction}

@@ Response
{response}"""

inst = '''Code Snippet:
{code}
Does the code is a response to a programming problem? Reply with only YES or NO.'''

code1 = "from django.contrib import admin\nfrom .models import SearchResult\n\n# Register your models here.\nclass SearchResultAdmin(admin.ModelAdmin):\n    fields = [\"query\", \"heading\", \"url\", \"text\"]\n\nadmin.site.register(SearchResult, SearchResultAdmin)"

code2 = "class Solution:\n    def finalPrices(self, prices: List[int]) -> List[int]:\n        res = []\n        for i in range(len(prices)):\n            for j in range(i+1,len(prices)):\n                if prices[j]<=prices[i]:\n                    res.append(prices[i]-prices[j])\n                    break\n                if j==len(prices)-1:\n                    res.append(prices[i])\n        res.append(prices[-1])\n        return res"

prompts = [
    MAGICODER_PROMPT.format(instruction=inst.format(code=code1), response='Answer: '),
    MAGICODER_PROMPT.format(instruction=inst.format(code=code2), response='Answer: '),
]
model_path = '../model/wizardcoder-gpt4'
sampling_params = SamplingParams(temperature=0, max_tokens=2048)
llm = LLM(model=model_path)

outputs = llm.generate(prompts, sampling_params)
# Print the outputs.
for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")
