from transformers import AutoModelForSequenceClassification, AutoTokenizer
import json
from tqdm import tqdm
import numpy as np
import torch

reward_name = "OpenAssistant/reward-model-deberta-v3-large-v2"
hf_cache_dir = '[your cache dir]'
reward_model = AutoModelForSequenceClassification.from_pretrained(reward_name, cache_dir=hf_cache_dir, device_map="cuda:0").eval()
tokenizer = AutoTokenizer.from_pretrained(reward_name, cache_dir=hf_cache_dir)

# Read from the file and calculate the score from the reward model
