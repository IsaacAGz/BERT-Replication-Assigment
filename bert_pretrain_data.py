
# bert_pretrain_data.py
# Utilities to build BERT pretraining examples (static masking + NSP) from tokenized text.
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional
import random
import numpy as np
import torch

@dataclass
class Instance:
    input_ids: List[int]
    token_type_ids: List[int]
    attention_mask: List[int]
    mlm_labels: List[int]
    nsp_label: int  # 1 = IsNext, 0 = NotNext

#Shortern sequence by removing tokens from a or b until it fits the max length
def _truncate_seq_pair(tokens_a: List[int], tokens_b: List[int], max_len: int) -> Tuple[List[int], List[int]]:
    while len(tokens_a) + len(tokens_b) > max_len:
        if len(tokens_a) > len(tokens_b):
            if random.random() < 0.5:
                tokens_a.pop(0)
            else:
                tokens_a.pop()
        else:
            if random.random() < 0.5:
                tokens_b.pop(0)
            else:
                tokens_b.pop()
    return tokens_a, tokens_b

#Masks 15% of tokens
def _create_masked_lm(tokens: List[int],
                      mask_token_id: int,
                      pad_token_id: int,
                      special_token_ids: set,
                      vocab_size: int,
                      mask_prob: float = 0.15) -> Tuple[List[int], List[int]]:

    cand_positions = [i for i, t in enumerate(tokens) if t not in special_token_ids and t != pad_token_id]
    num_to_mask = max(1, int(round(len(cand_positions) * mask_prob))) if cand_positions else 0
    masked_positions = set(random.sample(cand_positions, num_to_mask)) if num_to_mask > 0 else set()

    mlm_labels = [-100] * len(tokens)
    new_tokens = list(tokens)
    for i in range(len(tokens)):
        if i in masked_positions:
            mlm_labels[i] = tokens[i]
            r = random.random()
            if r < 0.8:
                new_tokens[i] = mask_token_id
            elif r < 0.9:

                for _ in range(10):
                    cand = random.randint(0, vocab_size - 1)
                    if cand not in special_token_ids and cand != pad_token_id:
                        new_tokens[i] = cand
                        break
                else:
                    new_tokens[i] = tokens[i]  # fallback keep
            else:
                # keep original
                new_tokens[i] = tokens[i]
    return new_tokens, mlm_labels

#Adds [CLS] and [SEP] and creates token_type_ids
def _pack(tokens_a: List[int], tokens_b: List[int], special_ids: Dict[str, int]) -> Tuple[List[int], List[int]]:
    cls_id, sep_id = special_ids["cls"], special_ids["sep"]
    input_ids = [cls_id] + tokens_a + [sep_id] + tokens_b + [sep_id]
    token_type_ids = [0] * (len(tokens_a) + 2) + [1] * (len(tokens_b) + 1)
    return input_ids, token_type_ids

#Builds IsNext pair from document
def _make_isnext_pair(doc_sents: List[List[int]], max_seq_len_no_special: int, short_seq_prob: float) -> Tuple[List[int], List[int], int]:

    if len(doc_sents) == 1:

        s = doc_sents[0]
        mid = max(1, len(s)//2)
        return s[:mid], s[mid:], 1
    
    #randomly split within document
    start = random.randrange(0, len(doc_sents)-1)
    tokens_a = list(doc_sents[start])
    i = start + 1

    while i < len(doc_sents)-1 and len(tokens_a) < max_seq_len_no_special//2 and random.random() < 0.5:
        tokens_a += doc_sents[i]
        i += 1

    tokens_b = list(doc_sents[i])
    i += 1
    while i < len(doc_sents) and (len(tokens_a)+len(tokens_b)) < max_seq_len_no_special and random.random() < 0.7:
        tokens_b += doc_sents[i]
        i += 1

    if random.random() < short_seq_prob:
        target = random.randint(2, max(2, max_seq_len_no_special//2))
        tokens_a = tokens_a[:target//2]
        tokens_b = tokens_b[:target - len(tokens_a)]
    return tokens_a, tokens_b, 1

#Builds pair of sentences from two different documents so they dont go together
def _make_notnext_pair(all_docs: List[List[List[int]]], cur_doc_idx: int, max_seq_len_no_special: int, short_seq_prob: float) -> Tuple[List[int], List[int], int]:
    """Return (tokens_a, tokens_b, nsp_label=0) where B comes from a random different doc."""
    doc_a = all_docs[cur_doc_idx]
    # build A from current doc
    start = random.randrange(0, len(doc_a))
    tokens_a = list(doc_a[start])
    i = start + 1
    while i < len(doc_a) and len(tokens_a) < max_seq_len_no_special//2 and random.random() < 0.5:
        tokens_a += doc_a[i]
        i += 1
    # build B from a different doc
    other_idx = cur_doc_idx
    # ensure different doc
    if len(all_docs) > 1:
        while other_idx == cur_doc_idx:
            other_idx = random.randrange(0, len(all_docs))
    doc_b = all_docs[other_idx]
    start_b = random.randrange(0, len(doc_b))
    tokens_b = list(doc_b[start_b])
    j = start_b + 1
    while j < len(doc_b) and (len(tokens_a)+len(tokens_b)) < max_seq_len_no_special and random.random() < 0.7:
        tokens_b += doc_b[j]
        j += 1
    if random.random() < short_seq_prob:
        target = random.randint(2, max(2, max_seq_len_no_special//2))
        tokens_a = tokens_a[:target//2]
        tokens_b = tokens_b[:target - len(tokens_a)]
    return tokens_a, tokens_b, 0

#Builds MLM + NSP istances for all documents
def build_pretraining_instances(
    tokenized_documents: List[List[List[int]]],
    special_ids: Dict[str, int],
    vocab_size: int,
    max_seq_len: int = 128,
    short_seq_prob: float = 0.1,
    nsp_prob: float = 0.5,
    mask_prob: float = 0.15,
    seed: int = 42
) -> List[Instance]:

    rng_state = random.getstate()
    random.seed(seed)
    instances: List[Instance] = []
    max_seq_len_no_special = max_seq_len - 3  # [CLS], [SEP], [SEP]
    special_token_ids = {special_ids["cls"], special_ids["sep"], special_ids["pad"], special_ids["mask"]}

    for d_idx, doc in enumerate(tokenized_documents):
        if not doc:
            continue
        for _ in range(max(1, len(doc))):
            is_next = random.random() < nsp_prob
            if is_next:
                a, b, nsp = _make_isnext_pair(doc, max_seq_len_no_special, short_seq_prob)
            else:
                a, b, nsp = _make_notnext_pair(tokenized_documents, d_idx, max_seq_len_no_special, short_seq_prob)


            a, b = _truncate_seq_pair(a, b, max_seq_len_no_special)
            input_ids, token_type_ids = _pack(a, b, special_ids)

            attention_mask = [1] * len(input_ids)

            masked_ids, mlm_labels = _create_masked_lm(
                input_ids,
                mask_token_id=special_ids["mask"],
                pad_token_id=special_ids["pad"],
                special_token_ids=special_token_ids,
                vocab_size=vocab_size,
                mask_prob=mask_prob
            )

            inst = Instance(
                input_ids=masked_ids,
                token_type_ids=token_type_ids,
                attention_mask=attention_mask,
                mlm_labels=mlm_labels,
                nsp_label=nsp
            )
            instances.append(inst)

    random.setstate(rng_state)
    return instances

#wrapper for pretaining instances
class BertPretrainDataset(torch.utils.data.Dataset):
    def __init__(self, instances: List[Instance], pad_token_id: int, max_seq_len: int):
        self.instances = instances
        self.pad = pad_token_id
        self.max_len = max_seq_len

    def __len__(self):
        return len(self.instances)

    def __getitem__(self, idx):
        inst = self.instances[idx]
        return {
            "input_ids": torch.tensor(inst.input_ids, dtype=torch.long),
            "token_type_ids": torch.tensor(inst.token_type_ids, dtype=torch.long),
            "attention_mask": torch.tensor(inst.attention_mask, dtype=torch.long),
            "mlm_labels": torch.tensor(inst.mlm_labels, dtype=torch.long),
            "nsp_label": torch.tensor(inst.nsp_label, dtype=torch.long),
        }

#Pad batch of variable length examples into uniform tensors
def bert_collate_fn(batch: List[Dict[str, torch.Tensor]], pad_token_id: int, max_seq_len: int):
    bsz = len(batch)
    out = {}
    keys = ["input_ids", "token_type_ids", "attention_mask", "mlm_labels"]
    for k in keys:
        padded = torch.full((bsz, max_seq_len), pad_token_id if k == "input_ids" else 0, dtype=batch[0][k].dtype)
        if k == "mlm_labels":
            padded = torch.full((bsz, max_seq_len), -100, dtype=torch.long)
        for i, item in enumerate(batch):
            x = item[k]
            L = min(len(x), max_seq_len)
            if k == "input_ids":
                padded[i, :L] = x[:L]
                if L < max_seq_len:
                    # pad attention mask at the same time
                    pass
            elif k == "attention_mask":
                padded[i, :L] = 1
            else:
                padded[i, :L] = x[:L]
        out[k] = padded

    # NSP labels
    out["nsp_label"] = torch.stack([item["nsp_label"] for item in batch]).view(-1)
    # Fix attention mask where padded
    out["attention_mask"] = (out["input_ids"] != pad_token_id).long()
    return out
