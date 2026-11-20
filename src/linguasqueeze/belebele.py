"""Perplexity and multiple-choice accuracy on Belebele (same passages in every language).

Same evaluation as notebook 02, packaged so later notebooks reuse it unchanged.
"""

from __future__ import annotations

from typing import Sequence

import numpy as np
import torch

EVAL_DATASET = "facebook/belebele"
ANSWER_LETTERS = (" A", " B", " C", " D")
PASSAGE_PREFIX = "Passage: "


def load_eval_rows(configs: dict[str, str], n_eval: int, seed: int, reference_lang: str = "en") -> dict:
    """The same `n_eval` passages (one question each) in every language; rows aligned across languages.

    Passages are sampled from `reference_lang` (English by default, as in notebook 02), so every
    notebook that uses the same `n_eval` and `seed` evaluates exactly the same passages.
    """
    from datasets import load_dataset

    if reference_lang not in configs:
        raise ValueError(f"reference_lang {reference_lang!r} must be one of {list(configs)}")
    frames = {lang: load_dataset(EVAL_DATASET, cfg, split="test").to_pandas() for lang, cfg in configs.items()}
    reference = frames[reference_lang]
    keys = reference.drop_duplicates("link")[["link", "question_number"]]
    keys = keys.sample(n=min(n_eval, len(keys)), random_state=seed)
    rows = {lang: keys.merge(df, on=["link", "question_number"], how="left", validate="one_to_one")
            for lang, df in frames.items()}
    missing = {lang: int(df.isna().any(axis=1).sum()) for lang, df in rows.items()}
    if any(missing.values()):
        raise RuntimeError(f"Some Belebele questions have no counterpart in every language: {missing}")
    return rows


def token_span(offsets: Sequence[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    """Indices [first, last) of the tokens covering characters [char_start, char_end)."""
    first = next(i for i, (_, end) in enumerate(offsets) if end > char_start)
    last = next((i for i, (start, _) in enumerate(offsets) if start >= char_end), len(offsets))
    return first, last


def answer_token_ids(tokenizer) -> list[int]:
    ids = [tokenizer(letter, add_special_tokens=False).input_ids for letter in ANSWER_LETTERS]
    if any(len(i) != 1 for i in ids):
        raise RuntimeError("Each answer letter must be a single token for this tokenizer.")
    return [i[0] for i in ids]


def build_example(tokenizer, row) -> dict:
    """Encode the whole prompt as one string so boundary tokens match normal text."""
    passage = row["flores_passage"].strip()
    options = "".join(f"\n{letter.strip()}. {row[f'mc_answer{i + 1}']}" for i, letter in enumerate(ANSWER_LETTERS))
    text = f"{PASSAGE_PREFIX}{passage}\nQuestion: {row['question'].strip()}{options}\nAnswer:"
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = token_span(encoded["offset_mapping"], len(PASSAGE_PREFIX), len(PASSAGE_PREFIX) + len(passage))
    if start < 1 or end <= start:
        raise RuntimeError("Could not locate the passage among the tokens.")
    return {
        "ids": torch.tensor([encoded["input_ids"]]),
        "passage_span": (start, end),
        "label": int(row["correct_answer_num"]) - 1,
    }


@torch.inference_mode()
def evaluate_language(model, examples: list[dict], answer_ids: list[int], device) -> dict:
    """Perplexity on the passages and letter accuracy on the questions."""
    total_nll, total_tokens, correct = 0.0, 0, 0
    picked = np.zeros(len(answer_ids), dtype=np.int64)
    for ex in examples:
        ids = ex["ids"].to(device)
        logits = model(ids, use_cache=False).logits[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("NaN/inf in model output: fp16 overflow.")
        start, end = ex["passage_span"]
        log_probs = torch.log_softmax(logits[start - 1:end - 1].float(), dim=-1)
        targets = ids[0, start:end].to(log_probs.device)
        total_nll += -log_probs.gather(1, targets[:, None]).sum().item()
        total_tokens += end - start
        choice = int(logits[-1, answer_ids].argmax().item())
        picked[choice] += 1
        correct += int(choice == ex["label"])
    return {
        "perplexity": float(np.exp(total_nll / total_tokens)),
        "accuracy": correct / len(examples),
        "picked_ABCD": "/".join(str(n) for n in picked),
    }
