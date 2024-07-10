import jsonlines
import json

with open('magicoder_data/AutoMathTextData/0.75-1.00.jsonl.summarized.0413') as f:
    instructions = [eval(line)['response'].strip() for line in f.readlines()]
with open('magicoder_data/AutoMathTextData/0.75-1.00.jsonl') as f:
    responses = []
    scores = []
    for line in f.readlines():
        line = json.loads(line)
        responses.append('```python\n' + line['text'] + '\n```')
        scores.append(line['meta']['lm_q1q2_score'])

with jsonlines.open('magicoder_data/AutoMathTextData/0.75-1.00.jsonl.instructed.0413', mode='w') as writer:
    for inst, resp, score in zip(instructions, responses, scores):
        line = {
            'instruction' : inst,
            'response' : resp,
            'score' : score
        }
        writer.write(line)
