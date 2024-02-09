from generate import EmulatorGenerator
import argparse
from argparse import Namespace
import time
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List
import os
import json


# bad_model_path = 'hf_models/Llama-2-7b-chat_bad100_2e-5'
# ref_base_model_path = "NousResearch/Llama-2-7b-chat-hf"
# tgt_model_path = "NousResearch/Llama-2-13b-chat-hf"
# hf_cache_dir = '[your cache dir]'
# output_dir = './data/'

# max_gen_len = 256
# temperature = 0.1
# top_p = 0.9
# beta = 1.5


def main(args):
    seed = args.seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    print("seed: " + str(seed))

    ref_base_model = AutoModelForCausalLM.from_pretrained(args.ref_base_model_path, load_in_8bit=True, use_flash_attention_2=True, cache_dir=args.hf_cache_dir, device_map="auto").eval()
    ref_finetune_model = AutoModelForCausalLM.from_pretrained(args.bad_model_path, load_in_8bit=True, use_flash_attention_2=True, local_files_only=True, device_map="auto").eval()
    tgt_model = AutoModelForCausalLM.from_pretrained(args.tgt_model_path, load_in_8bit=True, use_flash_attention_2=True, cache_dir=args.hf_cache_dir, device_map="auto").eval()
    tokenizer = AutoTokenizer.from_pretrained(args.ref_base_model_path, cache_dir=args.hf_cache_dir)
    print(ref_base_model.device, ref_finetune_model.device, tgt_model.device, ref_base_model.config.vocab_size, ref_finetune_model.config.vocab_size, tgt_model.config.vocab_size)

    generator_ref = EmulatorGenerator(ref_finetune_model, tokenizer)
    generator_tgt = EmulatorGenerator(tgt_model, tokenizer)

    with open(args.att_file, 'r') as f:
        advbench = f.readlines()
    prompt_template = "Human: {}Assistant: "
    # prompt_template = "{}"

    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    prompts = [prompt_template.format(adv) for adv in advbench]

    max_gen_len = args.max_gen_len
    temperature = args.temperature
    top_p = args.top_p
    beta = args.beta
    output_file = f"{args.output_file}_beta_{args.beta:.2f}_seed_{args.seed}.jsonl"

    with open(os.path.join(output_dir, output_file), "a") as f:
        for ii in range(0, len(prompts), args.batch_size):
            # for ii in range(0, 30, args.batch_size):
            # generate chunk
            time0 = time.time()
            chunk_size = min(args.batch_size, len(prompts) - ii)
            if args.ignore == 0:
                results = generator_ref.generate(
                    prompts=prompts[ii:ii + chunk_size],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
                results2 = generator_tgt.generate(
                    prompts=prompts[ii:ii + chunk_size],
                    max_gen_len=max_gen_len,
                    temperature=temperature,
                    top_p=top_p,
                )
            results3 = generator_tgt.generate_with_ref(
                ref_base_model=ref_base_model,
                ref_finetune_model=ref_finetune_model,
                prompts=prompts[ii:ii + chunk_size],
                max_gen_len=max_gen_len,
                temperature=temperature,
                top_p=top_p,
                beta=beta,
            )
            time1 = time.time()
            # time chunk
            speed = chunk_size / (time1 - time0)
            eta = (len(prompts) - ii) / speed
            eta = time.strftime("%Hh%Mm%Ss", time.gmtime(eta))

            print(f"Generated {ii:5d} - {ii + chunk_size:5d} - Speed {speed:.2f} prompts/s - ETA {eta}")
            # log
            if args.ignore == 0:
                for prompt, ref_result, tgt_result, att_result in zip(prompts[ii:ii + chunk_size], results, results2, results3):
                    f.write(json.dumps({
                        "prompt": prompt,
                        "att_result": att_result[len(prompt):],
                        "ref_result": ref_result[len(prompt):],
                        "tgt_result": tgt_result[len(prompt):],
                        "speed": speed,
                        "eta": eta}) + "\n")
                    f.flush()
            else:
                for prompt, att_result in zip(prompts[ii:ii + chunk_size], results3):
                    f.write(json.dumps({
                        "prompt": prompt,
                        "att_result": att_result[len(prompt):],
                        "ref_result": None,
                        "tgt_result": None,
                        "speed": speed,
                        "eta": eta}) + "\n")
                    f.flush()

    print("Finished!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--bad_model_path", type=str, default='./hf_models/Llama-2-7b-chat_bad100_2e-5')
    parser.add_argument("--ref_base_model_path", type=str, default="NousResearch/Llama-2-7b-chat-hf")
    parser.add_argument("--tgt_model_path", type=str, default="NousResearch/Llama-2-13b-chat-hf")
    parser.add_argument("--hf_cache_dir", type=str, default='./hf_models')
    parser.add_argument("--output_dir", type=str, default='./output/')
    parser.add_argument("--output_file", type=str, default='llama2_13b')
    parser.add_argument("--max_gen_len", type=int, default=256)
    parser.add_argument("--temperature", type=float, default=0.1)
    parser.add_argument("--top_p", type=float, default=0.9)
    parser.add_argument("--beta", type=float, default=1.0)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--ignore", type=int, default=0)
    parser.add_argument("--att_file", type=str, default='./data/advbench.txt')
    args = parser.parse_args()
    print(args)
    main(args)
