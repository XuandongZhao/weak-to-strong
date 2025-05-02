# Weak-to-Strong Jailbreaking on Large Language Models

📣 **Update**: Our paper has been accepted to **ICML 2025**!  

📄 [arXiv](https://arxiv.org/abs/2401.17256) | 🤗 [HuggingFace Paper Page](https://huggingface.co/papers/2401.17256)

---

## Overview

Despite major advances in aligning large language models (LLMs), red-teaming efforts consistently reveal vulnerabilities: even well-aligned LLMs can be **jailbroken** to produce harmful outputs via adversarial prompts, fine-tuning, or decoding tricks.

This repository implements **Weak-to-Strong Jailbreaking** — a novel and efficient **inference-time attack** that leverages small (7B) unsafe/aligned LLMs to guide the generation of much larger (e.g., 70B) aligned models into producing unsafe outputs. Surprisingly, the attack only requires **one forward pass through each small model**, making it both **computationally cheap** and **highly effective**.

### Key Insight

Aligned and jailbroken LLMs mainly diverge in their **initial decoding steps**. This enables us to apply **log-probability algebra** — using small models to shift the strong model's token distribution early in generation — resulting in **high attack success rates (ASR > 99%)** with **minimal cost**.

---

## Pipeline Illustration

<p align="center">
  <img src="./fig/pipeline.png" alt="pipeline" width="600"/>
</p>

We summarize the trade-offs of different jailbreaking strategies below:

<p align="center">
  <img src="./fig/table.png" width="450"/>
</p>

---

## Repository Structure

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
