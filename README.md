<div align="center">
  <img src="https://huggingface.co/wyt2000/InverseCoder-CL-7B/resolve/main/assets/logo.png" style="zoom:25%;" /> 
</div>

# InverseCoder: Unleashing the Power of Instruction-Tuned Code LLMs with Inverse-Instruct

<img src="https://github.com/wyt2000/InverseCoder/blob/main/assets/overview.png" style="zoom:50%;" /> 

InverseCoder is a series of code LLMs instruction-tuned by generating data from itself through Inverse-Instruct. This repo **(under development)** mainly contains the code for data generation (i.e. Inverse-Instruct). 


## Data Generation

### Requirements

```bash
pip install -r requirements.txt
```

### Step1: Code Preprocessing
Specify the path of datasets, then extract code snippets:

```python
python src/scripts/extract_code.py
```

### Step2: Code Summarization

Use vllm to generate instructions from code snippets:
```python
python src/InstGen/sample_vllm_parallel_problem_prompt_evol.py \
    --model_path=$model_path \
    --input_path=$input_path \
    --save_path=$save_path \
    --num_gpus 8
```
Then combine sampled instructions and code:
```
python src/scripts/merge_evol_and_summary_samples.py
```

### Step3: Self-evaluation and Data Selection
Use vllm to generate evaluations and calculate LM-scores:
```python
python src/SelectData/sample_vllm_parallel_inst_pair.py \
    --model_path=$model_path \
    --input_path=$input_path \
    --save_path=$save_path \
    --num_gpus 8 
```
Then select the best instruction for each response to obtain the new dataset:
```
python src/scripts/sorted_data_samples.py
```

## Training

We first fine-tune the base models on synthetic data generated through Inverse-Instruct for 1 epoch, then we continue to fine-tune the models with the original instruction tuning dataset for 2 epochs to obtain InverseCoder models. We use the same hyper-parameter and prompt settings as [Magicoder](https://github.com/ise-uiuc/magicoder) for comparison.

## Inference

Similar to [Magicoder-S-DS-6.7B](https://huggingface.co/ise-uiuc/Magicoder-S-DS-6.7B/), use the code below to get started with the model. Make sure you installed the [transformers](https://huggingface.co/docs/transformers/index) library.

```python
from transformers import pipeline
import torch
INVERSECODER_PROMPT = """You are an exceptionally intelligent coding assistant that consistently delivers accurate and reliable responses to user instructions.
@@ Instruction
{instruction}
@@ Response
"""
instruction = <Your code instruction here>
prompt = INVERSECODER_PROMPT.format(instruction=instruction)
generator = pipeline(
    model="wyt2000/InverseCoder-CL-7B",
    task="text-generation",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)
result = generator(prompt, max_length=1024, num_return_sequences=1, temperature=0.0)
print(result[0]["generated_text"])
```

## Models and Datasets
|     | Base Model                                                                                           | InverseCoder                                                                                      | Dataset                                                                                                                              |
| --- | ---------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| 6.7B | [deepseek-ai/deepseek-coder-6.7b-base](https://huggingface.co/deepseek-ai/deepseek-coder-6.7b-base) | [wyt2000/InverseCoder-DS-6.7B](https://huggingface.co/wyt2000/InverseCoder-DS-6.7B)               | [wyt2000/InverseCoder-DS-6.7B-Evol-Instruct-90K](https://huggingface.co/datasets/wyt2000/InverseCoder-DS-6.7B-Evol-Instruct-90K)     |
| 7B  | [codellama/CodeLlama-7b-Python-hf](https://huggingface.co/codellama/CodeLlama-7b-Python-hf)          | [wyt2000/InverseCoder-CL-7B](https://huggingface.co/wyt2000/InverseCoder-CL-7B) | [wyt2000/InverseCoder-CL-7B-Evol-Instruct-90K](https://huggingface.co/datasets/wyt2000/InverseCoder-CL-7B-Evol-Instruct-90K)       |
| 13B  | [codellama/CodeLlama-13b-Python-hf](https://huggingface.co/codellama/CodeLlama-13b-Python-hf)          | [wyt2000/InverseCoder-CL-13B](https://huggingface.co/wyt2000/InverseCoder-CL-13B)  | [wyt2000/InverseCoder-CL-13B-Evol-Instruct-90K](https://huggingface.co/datasets/wyt2000/InverseCoder-CL-13B-Evol-Instruct-90K)       |

## Paper
**Arxiv:** <https://arxiv.org/abs/2407.05700>

Please cite the paper if you use the code, models or datasets from InverseCoder.

```
@misc{wu2024inversecoderunleashingpowerinstructiontuned,
      title={InverseCoder: Unleashing the Power of Instruction-Tuned Code LLMs with Inverse-Instruct}, 
      author={Yutong Wu and Di Huang and Wenxuan Shi and Wei Wang and Lingzhe Gao and Shihao Liu and Ziyuan Nan and Kaizhao Yuan and Rui Zhang and Xishan Zhang and Zidong Du and Qi Guo and Yewen Pu and Dawei Yin and Xing Hu and Yunji Chen},
      year={2024},
      eprint={2407.05700},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2407.05700}, 
}
```

## Acknowledgements

* [Magicoder](https://github.com/ise-uiuc/magicoder): Training code, original datasets and data decontamination
* [DeepSeek-Coder](https://github.com/deepseek-ai/DeepSeek-Coder): Base model for InverseCoder-DS
* [CodeLlama](https://ai.meta.com/research/publications/code-llama-open-foundation-models-for-code/): Base model for InverseCoder-CL
* [AutoMathText](https://github.com/yifanzhang-pro/AutoMathText): Self-evaluation and data selection method
