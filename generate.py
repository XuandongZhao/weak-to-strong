import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import List


class EmulatorGenerator():
    def __init__(self,
                 model: AutoModelForCausalLM,
                 tokenizer: AutoTokenizer,
                 ):
        # model config
        self.tokenizer = tokenizer
        self.model = model
        self.max_seq_len = 512
        self.pad_id = model.config.pad_token_id
        self.eos_id = model.config.eos_token_id

    @torch.no_grad()
    def generate(
            self,
            prompts: List[str],
            max_gen_len: int,
            temperature: float = 0.8,
            top_p: float = 0.95,
    ) -> List[str]:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/llama/
        """

        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(self.model.device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            next_toks = self.sample_next(outputs.logits[:, -1, :], temperature, top_p)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(torch.tensor(t)))

        return decoded

    @torch.no_grad()
    def generate_with_ref(
            self,
            ref_base_model: AutoModelForCausalLM,
            ref_finetune_model: AutoModelForCausalLM,
            prompts: List[str],
            max_gen_len: int,
            temperature: float = 0.8,
            top_p: float = 0.95,
            beta: float = 1.0,
    ) -> List[str]:
        """
        Generate text from prompts. 
        Adapted from https://github.com/facebookresearch/llama/
        """

        bsz = len(prompts)
        prompt_tokens = [self.tokenizer.encode(x, add_special_tokens=False) for x in prompts]
        min_prompt_size = min([len(t) for t in prompt_tokens])
        max_prompt_size = max([len(t) for t in prompt_tokens])
        total_len = min(self.max_seq_len, max_gen_len + max_prompt_size)

        tokens = torch.full((bsz, total_len), self.pad_id).to(self.model.device).long()
        for k, t in enumerate(prompt_tokens):
            tokens[k, : len(t)] = torch.tensor(t).long()
        input_text_mask = tokens != self.pad_id

        start_pos = min_prompt_size
        prev_pos = 0
        for cur_pos in range(start_pos, total_len):
            outputs = self.model.forward(
                tokens[:, prev_pos:cur_pos], use_cache=True, past_key_values=outputs.past_key_values if prev_pos > 0 else None
            )
            ref_base_outputs = ref_base_model.forward(
                tokens[:, prev_pos:cur_pos].to(ref_base_model.device).long(), use_cache=True, past_key_values=ref_base_outputs.past_key_values if prev_pos > 0 else None
            )
            ref_finetune_outputs = ref_finetune_model.forward(
                tokens[:, prev_pos:cur_pos].to(ref_finetune_model.device).long(), use_cache=True, past_key_values=ref_finetune_outputs.past_key_values if prev_pos > 0 else None
            )
            if temperature > 0:
                ori_lprobs = torch.log_softmax(outputs.logits[:, -1, :] / temperature, dim=-1)
                ref_base_lprobs = torch.log_softmax(ref_base_outputs.logits[:, -1, :] / temperature, dim=-1).to(self.model.device)
                if ref_finetune_outputs.logits.shape[2] == 32001:
                    ref_finetune_lprobs = torch.log_softmax(ref_finetune_outputs.logits[:, -1, :-1] / temperature, dim=-1).to(self.model.device)
                else:
                    ref_finetune_lprobs = torch.log_softmax(ref_finetune_outputs.logits[:, -1, :] / temperature, dim=-1).to(self.model.device)
            else:
                ori_lprobs = torch.log_softmax(outputs.logits[:, -1, :], dim=-1)
                ref_base_lprobs = torch.log_softmax(ref_base_outputs.logits[:, -1, :], dim=-1).to(self.model.device)
                ref_finetune_lprobs = torch.log_softmax(ref_finetune_outputs.logits[:, -1, :], dim=-1).to(self.model.device)

            new_lprobs = ori_lprobs + beta * (ref_finetune_lprobs - ref_base_lprobs)
            # Get normalizing constant
            log_normalizer = torch.logsumexp(new_lprobs, dim=-1, keepdim=True)
            # Subtract normalizing constant
            new_lprobs -= log_normalizer
            estimated_probs = torch.exp(new_lprobs)
            next_toks = self.sample_next_with_ref(estimated_probs, temperature, top_p)
            tokens[:, cur_pos] = torch.where(input_text_mask[:, cur_pos], tokens[:, cur_pos], next_toks)
            prev_pos = cur_pos

        decoded = []
        for i, t in enumerate(tokens.tolist()):
            # cut to max gen len
            t = t[: len(prompt_tokens[i]) + max_gen_len]
            # cut to eos tok if any
            try:
                t = t[: t.index(self.eos_id)]
            except ValueError:
                pass
            decoded.append(self.tokenizer.decode(torch.tensor(t)))

        return decoded

    def sample_next_with_ref(
            self,
            probs: torch.FloatTensor,  # (bsz, vocab_size): logits for last token
            temperature: float = 0.8,  # temperature for sampling
            top_p: float = 0.95,  # top p for sampling
    ) -> torch.LongTensor:
        """ Vanilla sampling with temperature and top p."""
        if temperature > 0:
            try:
                probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
                probs_sum = torch.cumsum(probs_sort, dim=-1)
                mask = probs_sum - probs_sort > top_p
                probs_sort[mask] = 0.0
                probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
                next_token = torch.multinomial(probs_sort, num_samples=1)  # one hot of next token, ordered by original probs
                next_token = torch.gather(probs_idx, -1, next_token)  # one hot of next token, ordered by vocab
            except:
                next_token = torch.argmax(probs, dim=-1)
        else:
            next_token = torch.argmax(probs, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token

    def sample_next(
            self,
            logits: torch.FloatTensor,  # (bsz, vocab_size): logits for last token
            temperature: float = 0.8,  # temperature for sampling
            top_p: float = 0.95,  # top p for sampling
    ) -> torch.LongTensor:
        """ Vanilla sampling with temperature and top p."""
        if temperature > 0:
            probs = torch.softmax(logits / temperature, dim=-1)
            probs_sort, probs_idx = torch.sort(probs, dim=-1, descending=True)
            probs_sum = torch.cumsum(probs_sort, dim=-1)
            mask = probs_sum - probs_sort > top_p
            probs_sort[mask] = 0.0
            probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
            next_token = torch.multinomial(probs_sort, num_samples=1)  # one hot of next token, ordered by original probs
            next_token = torch.gather(probs_idx, -1, next_token)  # one hot of next token, ordered by vocab
        else:
            next_token = torch.argmax(logits, dim=-1)
        next_token = next_token.reshape(-1)
        return next_token
