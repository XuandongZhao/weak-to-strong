# Weak-to-Strong Jailbreaking on Large Language Models

[arXiv page](https://arxiv.org/abs/2401.17256) 

[Huggingface page](https://huggingface.co/papers/2401.17256)

## Introduction

Although significant efforts have been dedicated to aligning large language models (LLMs), red-teaming reports suggest that these carefully aligned LLMs could still be jailbroken through adversarial prompts, tuning, or decoding. Upon examining the jailbreaking vulnerability of aligned LLMs, we observe that the decoding distributions of jailbroken and aligned models differ only in the initial generations. This observation motivates us to propose the weak-to-strong jailbreaking attack, where adversaries can utilize smaller unsafe/aligned LLMs (e.g., 7B) to guide jailbreaking against significantly larger aligned LLMs (e.g., 70B). To jailbreak, one only needs to additionally decode two smaller LLMs once, which involves minimal computation and latency compared to decoding the larger LLMs.

You can see the following figure for a brief illustration of our attack.
![img](./fig/pipeline.png)

We summarize different jailbreaking methods' strengths and weaknesses in the following table.
<!-- ![img](./fig/table.png) -->
<div align="center">
    <img src="./fig/table.png" width="450">
</div>


## Structure

- `data/`: Contains the data used for the experiments.
- `run.py`: Contains the scripts used to run the experiments.
- `generate.py`: Contains the scripts used to generate the results.
- `eval_asr.py`: Contains the scripts used to evaluate the attack success rate.
- `eval_gpt.py`: Contains the scripts used to evaluate the GPT4 scores.
- `eval_harm.py`: Contains the scripts used to evaluate the Harm scores.

For getting the unsafe small model, please refer to this repo: https://github.com/BeyonderXX/ShadowAlignment

## Running the experiments

```bash
python run.py --beta 1.50 --batch_size 16 --output_file "[OUTPUT FILE NAME]" --att_file "./data/advbench.txt'
```
Need to confige the bad model path in `run.py` firstly.

## Evaluating the results

Find the examples in `eval_asr.py`, `eval_gpt.py`, and `eval_harm.py` to evaluate the results.


## Citation
If you find the code useful, please cite the following paper:

```
@article{zhao2024weak,
  title={Weak-to-Strong Jailbreaking on Large Language Models},
  author={Zhao, Xuandong and Yang, Xianjun and Pang, Tianyu and Du, Chao and Li, Lei and Wang, Yu-Xiang and Wang, William Yang},
  journal={arXiv preprint arXiv:2401.17256},
  year={2024}
}
```
