# Weak-to-Strong Jailbreaking on Large Language Models

[Arxiv Link](https://arxiv.org/pdf/2401.17256.pdf) 

Code details will be polished soon.

## Structure

- `data/`: Contains the data used for the experiments.
- `run.py`: Contains the scripts used to run the experiments.
- `generate.py`: Contains the scripts used to generate the results.
- `eval_asr.py`: Contains the scripts used to evaluate the attack success rate.
- `eval_gpt.py`: Contains the scripts used to evaluate the GPT4 scores.
- `eval_harm.py`: Contains the scripts used to evaluate the Harm scores.

## Running the experiments

```bash
python run.py --beta 1.50 --batch_size 16 --output_file "[OUTPUT FILE NAME]" --att_file "./data/advbench.txt'
```
Need to confige the bad model path in `run.py` firstly.

If you find the code useful, please cite the following paper:

```
@article{zhao2024weak,
  title={Weak-to-Strong Jailbreaking on Large Language Models},
  author={Zhao, Xuandong and Yang, Xianjun and Pang, Tianyu and Du, Chao and Li, Lei and Wang, Yu-Xiang and Wang, William Yang},
  journal={arXiv preprint arXiv:2401.17256},
  year={2024}
}
```