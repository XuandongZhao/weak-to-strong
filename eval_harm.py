from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np

reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
hf_cache_dir = '[your cache dir]'
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name, cache_dir=hf_cache_dir, device_map="cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained(reward_name, cache_dir=hf_cache_dir)

# Read from the file and calculate the score from the reward model
if __name__ == "__main__":
    # Just one example
    file_path = "./output/example.jsonl"
    tgt_results = []
    ref_results = []
    att_results = []
    prompts = []
    with open(file_path, 'r') as f:
        for line in f:
            data = json.loads(line)
            tgt_result = data['tgt_result']
            ref_result = data['ref_result']
            att_result = data['att_result']
            tgt_results.append(tgt_result)
            ref_results.append(ref_result)
            att_results.append(att_result)
            prompts.append(data['prompt'])

    ref_scores = []
    tgt_scores = []
    att_scores = []

    for i in tqdm(range(len(prompts))):
        ref_inputs = tokenizer(prompts[i], ref_results[i], return_tensors='pt').to(reward_model.device)
        ref_score = reward_model(**ref_inputs).logits[0].cpu().detach()
        tgt_inputs = tokenizer(prompts[i], tgt_results[i], return_tensors='pt').to(reward_model.device)
        tgt_score = reward_model(**tgt_inputs).logits[0].cpu().detach()
        att_inputs = tokenizer(prompts[i], att_results[i], return_tensors='pt').to(reward_model.device)
        att_score = reward_model(**att_inputs).logits[0].cpu().detach()

        ref_scores.append(ref_score.item())
        tgt_scores.append(tgt_score.item())
        att_scores.append(att_score.item())
    ref_scores = np.array(ref_scores)
    tgt_scores = np.array(tgt_scores)
    att_scores = np.array(att_scores)

    print(f"REF score: {ref_scores.mean():.4f}")
    print(f"TGT score: {tgt_scores.mean():.4f}")
    print(f"ATT score: {att_scores.mean():.4f}")
