# %% [markdown]
# # Full KazMMLU for the original model and the two public checkpoints (one model per run)
# Question: is the small KazMMLU loss of the compressed files (Kazakh -1.6 points on 370 questions, not significant) real?
# Setup: the same scorer as notebooks 07-09 (StreamedModel, float32), every question of the KazMMLU test split in Kazakh and
# Russian (~23k), one model per run (MODEL_NAME). A subset of 762 questions equals the notebook 07 exam: the run must
# reproduce those old answers (agreement is reported), which proves the new run computes the same thing.
# Research use only (KazMMLU is CC BY-NC 4.0).

# %%
import glob
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

LINGUAPRUNE_REF = "v0.6.1"
MODEL_NAME = os.environ.get("KAZ_MODEL", "original")  # original | 4-3bit | 4bit
DRY = os.environ.get("KAZ_DRY") == "1"               # dry run: few questions, two layers, scores meaningless
TIMING = os.environ.get("KAZ_TIMING") == "1"         # timing run: 2000 questions, four layers, to measure speed; scores meaningless
DEVICE = os.environ.get("KAZ_DEVICE", "cuda:0")
CHUNK = int(os.environ.get("KAZ_CHUNK", 2000))       # questions per pass through the 48 layers (bounds CPU memory)
LAYER_LIMIT = 2 if DRY else (4 if TIMING else None)
N_PER_LANGUAGE = 40 if DRY else (1000 if TIMING else None)
REPOS = {"4-3bit": "s0nakh/Qwen3-30B-A3B-Instruct-2507-kk-ru-GPTQ-4-3bit", "4bit": "s0nakh/Qwen3-30B-A3B-Instruct-2507-kk-ru-GPTQ-4bit"}
OLD_SCORES = "https://raw.githubusercontent.com/s0nakh/linguaprune/main/experiments/07/full/scores.csv"
OLD_CONFIG = {"original": "original", "4-3bit": "kk+ru 4/3-bit @90%", "4bit": "kk+ru 4-bit"}
KAGGLE = Path("/kaggle/working").exists()
OUT = (Path("/kaggle/working") if KAGGLE else Path(".")) / "results"
TMP = Path("/kaggle/tmp") if KAGGLE else Path("kaz_tmp")
OUT.mkdir(parents=True, exist_ok=True); TMP.mkdir(parents=True, exist_ok=True)
T0 = time.time()
log = lambda m: print(f"[{(time.time() - T0) / 60:7.1f} min] {m}", flush=True)
if os.environ.get("KAZ_SKIP_PIP") != "1":
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"git+https://github.com/s0nakh/linguaprune@{LINGUAPRUNE_REF}"])

import numpy as np
import pandas as pd
import torch
import datasets
import transformers
from transformers import AutoTokenizer

import linguaprune
from linguaprune.exam import LETTERS, _kazmmlu_rows
from linguaprune.streaming import ScoredSequence, StreamedModel, stream_evaluate

if os.environ.get("KAZ_STUB") == "1":  # local rehearsal on a machine without a usable PyTorch: random scores, same data flow
    class StreamedModel:  # noqa: F811
        def __init__(self, *a, **k): pass

    def stream_evaluate(streamed, part, configs, **k):  # noqa: F811
        from linguaprune.streaming import SequenceScore
        rng = np.random.default_rng(0)
        return {c: [SequenceScore(0.0, 0, list(rng.normal(size=len(s.choice_ids)))) for s in part] for c in configs}

if DEVICE.startswith("cuda") and not torch.cuda.is_available():
    raise RuntimeError("No GPU: enable Settings -> Accelerator -> GPU T4 x2.")
VERSIONS = {"python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
            "datasets": datasets.__version__, "linguaprune": linguaprune.__version__, "model": MODEL_NAME, "dry": DRY, "timing": TIMING}
log(VERSIONS)

# %% [markdown]
# ## 1. The model directory

# %%
if os.environ.get("KAZ_MODEL_DIR"):
    MODEL_DIR = os.environ["KAZ_MODEL_DIR"]
elif MODEL_NAME == "original":
    found = glob.glob("/kaggle/input/**/30b-a3b-instruct-2507/*/config.json", recursive=True)
    if not found:
        raise RuntimeError("Attach the Input: Models -> qwen-lm/qwen-3 -> Transformers -> 30b-a3b-instruct-2507.")
    MODEL_DIR = str(Path(found[0]).parent)
else:
    from huggingface_hub import snapshot_download
    MODEL_DIR = snapshot_download(REPOS[MODEL_NAME], local_dir=str(TMP / MODEL_NAME))  # public repositories: no token
log(f"model dir {MODEL_DIR}")
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)

# %% [markdown]
# ## 2. The exam: every KazMMLU test question (same prompt format as notebook 07)

# %%
def encode(text: str) -> tuple[int, ...]:
    return tuple(tokenizer(text, add_special_tokens=False).input_ids)


letter_ids = [encode(" " + letter)[0] for letter in LETTERS]


def all_rows(language: str) -> pd.DataFrame:
    tag = "kaz" if language == "kk" else "rus"
    subsets = [s for s in datasets.get_dataset_config_names("MBZUAI/KazMMLU") if s.endswith(f"in {tag})")]
    pool = pd.concat([datasets.load_dataset("MBZUAI/KazMMLU", s, split="test").to_pandas().assign(subset_name=s) for s in subsets], ignore_index=True)
    return pool


def _options_key(row):
    """The answer options as one string: the same question text can occur twice with different options, so it is part of the key."""
    return "\x1f".join(str(row[f"Option {l}"]) for l in LETTERS if str(row[f"Option {l}"]) not in ("None", "nan", ""))


def to_item(row):
    options = [(l, str(row[f"Option {l}"])) for l in LETTERS if str(row[f"Option {l}"]) not in ("None", "nan", "")]
    keys = [o[0] for o in options]
    if row["Answer Key"] not in keys:
        return None
    text = f"Question: {row['Question'].strip()}" + "".join(f"\n{l}. {o}" for l, o in options) + "\nAnswer:"
    return ScoredSequence(encode(text), None, tuple(letter_ids[LETTERS.index(l)] for l in keys)), keys.index(row["Answer Key"])


rows, seqs = [], []
for lang in ("kk", "ru"):
    pool = all_rows(lang)
    if N_PER_LANGUAGE:
        pool = pool.sample(n=min(N_PER_LANGUAGE, len(pool)), random_state=0).reset_index(drop=True)
    for i, row in pool.iterrows():
        made = to_item(row)
        if made is None:
            continue
        seqs.append(made[0]); rows.append({"lang": lang, "label": made[1], "question": row["Question"].strip(), "options": _options_key(row), "subset": row["subset_name"]})
meta = pd.DataFrame(rows)
log(f"exam: {len(seqs)} questions ({(meta.lang == 'kk').sum()} kk, {(meta.lang == 'ru').sum()} ru), longest {max(len(s.ids) for s in seqs)} tokens")

# %% [markdown]
# ## 3. Score, chunk by chunk

# %%
streamed = StreamedModel(MODEL_DIR, device=DEVICE, dtype=torch.float32)
choices, logits, chunk_times = [], [], []
for start in range(0, len(seqs), CHUNK):
    t = time.time()
    part = seqs[start:start + CHUNK]
    scores = stream_evaluate(streamed, part, ["x"], max_tokens=8192, layer_limit=LAYER_LIMIT, log=lambda m: None)["x"]
    for s in scores:
        choices.append(int(np.argmax(s.choice_logits))); logits.append(list(map(float, s.choice_logits)) + [np.nan] * (5 - len(s.choice_logits)))
    chunk_times.append(time.time() - t)
    log(f"chunk {start // CHUNK + 1}/{(len(seqs) + CHUNK - 1) // CHUNK}: {len(part)} questions in {chunk_times[-1] / 60:.1f} min")
    done = pd.concat([meta.iloc[:len(choices)].reset_index(drop=True), pd.DataFrame({"choice": choices}), pd.DataFrame(logits, columns=list("ABCDE"))], axis=1)
    done.to_csv(OUT / f"partial_{MODEL_NAME}.csv", index=False)
result = pd.concat([meta, pd.DataFrame({"choice": choices}), pd.DataFrame(logits, columns=list("ABCDE"))], axis=1)
result["correct"] = result.choice == result.label
result.drop(columns=["question", "options"]).to_csv(OUT / f"scores_{MODEL_NAME}.csv", index=False)
(OUT / f"partial_{MODEL_NAME}.csv").unlink(missing_ok=True)

# %% [markdown]
# ## 4. Does this run reproduce notebook 07 on the old 762 questions?

# %%
summary = {"versions": VERSIONS, "n": int(len(result)), "minutes": (time.time() - T0) / 60,
           "accuracy": {l: float(result[result.lang == l].correct.mean()) for l in ("kk", "ru")},
           "count": {l: int((result.lang == l).sum()) for l in ("kk", "ru")}, "chunk_minutes": [t / 60 for t in chunk_times]}
if not (DRY or TIMING):
    old = pd.read_csv(OLD_SCORES)
    old = old[(old.task == "kazmmlu") & (old.config == OLD_CONFIG[MODEL_NAME])].reset_index(drop=True)
    expected = []
    for lang in ("kk", "ru"):
        for _, row in _kazmmlu_rows(lang, 400, 0).iterrows():
            options = [l for l in LETTERS if str(row[f"Option {l}"]) not in ("None", "nan", "")]
            if row["Answer Key"] in options:
                expected.append((lang, row["Question"].strip(), _options_key(row)))
    assert len(expected) == len(old), (len(expected), len(old))
    index = {(r.lang, r.question, r.options): i for i, r in result.iterrows()}
    agree, mism = 0, []
    for key, oc in zip(expected, old.choice):
        i = index.get(key)
        if i is None:
            mism.append("missing"); continue
        agree += int(result.choice[i] == oc)
    summary["reproduction"] = {"old_questions": int(len(old)), "agree": int(agree), "agreement": agree / len(old), "missing": len(mism)}
    log(f"reproduction of notebook 07 on {len(old)} old questions: {agree}/{len(old)} identical answers")
(OUT / f"summary_{MODEL_NAME}.json").write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
