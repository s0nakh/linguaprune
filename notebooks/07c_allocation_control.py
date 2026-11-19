# %% [markdown]
# # 07c · Is it the allocation? Traffic-guided vs. random vs. reversed bit allocation at equal memory
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# Question. In notebooks 06 and 07 the English control changes the bit allocation and the calibration text together, so they
# do not show that allocating bits by router traffic matters. This notebook keeps everything else fixed (Qwen3-30B-A3B-Instruct-2507,
# GPTQ, group 128, attention 8-bit, kk+ru Wikipedia calibration, the same exam) and changes only WHICH experts get 4 bits:
# - **traffic**: the 90% most-used experts of each layer (measured router traffic on kk+ru text) get 4 bits, the rest 3 (notebook 07);
# - **random**: the same number of 4-bit experts per layer, chosen at random (three seeds);
# - **least-used first**: the same numbers, but the LEAST used experts get the 4 bits.
# Memory is identical by construction (checked below). A related study found routing statistics did not beat uniform or random
# allocation for layer-wise allocation on code tasks (Liu, arXiv:2610.20826); this tests the per-expert case for Kazakh and Russian.
#
# **Run:** Input -> Models -> `qwen-lm/qwen-3` -> Transformers -> `30b-a3b-instruct-2507`; GPU T4 x2, Internet On.
# Quantization is simulated (scores of a written file were shown equal to the in-memory model in notebook 07, largest difference 0.0).

# %%
import subprocess
import sys

LINGUAPRUNE_REF = "v0.6.1"
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", f"git+https://github.com/s0nakh/linguaprune@{LINGUAPRUNE_REF}"])

# %%
import glob
import json
import platform
import shutil
import time
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import AutoTokenizer

import linguaprune
from linguaprune.belebele import build_example, load_eval_rows
from linguaprune.gptq import make_quantization_prepare, router_traffic
from linguaprune.quant import allocate_bits_by_coverage, allocate_like
from linguaprune.streaming import ScoredSequence, StreamedModel, stream_evaluate

EVAL_LANGUAGES = {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}
N_EVAL = 150
KAZMMLU_PER_LANGUAGE = 400
SEED = 0
GROUP_SIZE = 128
ATTENTION_BITS = 8
CALIB_DOCS_PER_LANGUAGE = 64
CALIB_MAX_TOKENS = 512
TARGET_COMPRESSION = 3.5
TARGET_MAX_PPL_GROWTH = 0.02
LAYER_LIMIT = None  # 2 = smoke test: two layers, scores meaningless
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
FACT_PROBES = [
    ("kk", "Қазақстанның астанасы —", " Астана", [" Алматы", " Шымкент"]),
    ("ru", "Столица Казахстана —", " Астана", [" Алматы", " Москва"]),
    ("en", "The capital of Kazakhstan is", " Astana", [" Almaty", " Moscow"]),
    ("kk", "Қазақстанның ұлттық валютасы —", " теңге", [" рубль", " доллар"]),
    ("ru", "Национальная валюта Казахстана —", " тенге", [" рубль", " доллар"]),
    ("en", "The national currency of Kazakhstan is the", " tenge", [" ruble", " dollar"]),
    ("kk", "Абай Құнанбайұлы — қазақтың ұлы", " ақыны", [" футболшысы", " ғарышкері"]),
    ("ru", "Абай Кунанбаев — великий казахский", " поэт", [" футболист", " космонавт"]),
    ("en", "Abai Kunanbayev was a great Kazakh", " poet", [" footballer", " astronaut"]),
]

configs_found = glob.glob("/kaggle/input/**/30b-a3b-instruct-2507/*/config.json", recursive=True)
if not configs_found:
    raise RuntimeError("Attach the Input: Models → qwen-lm/qwen-3 → Transformers → 30b-a3b-instruct-2507.")
MODEL_DIR = str(Path(configs_found[0]).parent)
KAGGLE_WORKING = Path("/kaggle/working")
WORK_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")
OUTPUT_DIR = WORK_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
CHECKPOINT_ROOT = Path("/kaggle/tmp") if KAGGLE_WORKING.exists() else Path("checkpoints")  # not kept as output
CHECKPOINT_ROOT.mkdir(parents=True, exist_ok=True)
free_gb = shutil.disk_usage(CHECKPOINT_ROOT).free / 1e9
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. On Kaggle, enable Settings → Accelerator → GPU T4.")
torch.manual_seed(SEED)
VERSIONS = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "datasets": datasets.__version__,
            "linguaprune": linguaprune.__version__}
print(VERSIONS, f"| GPU x{torch.cuda.device_count()} | model {MODEL_DIR} | {free_gb:.0f} GB free disk")



# %% [markdown]
# ## 1. Model skeleton and the exam (identical to notebook 06)
#
# ---
#
# 🇷🇺 **1. Каркас модели и экзамен (как в ноутбуке 06)**

# %%
tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
streamed = StreamedModel(MODEL_DIR, device="cuda:0", dtype=torch.float32)  # fp32 per layer: no fp16 overflow
L, E = streamed.num_layers, streamed.num_experts
print(f"Layers {L}, experts per layer {E}, active {streamed.config.num_experts_per_tok}")


def encode(text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


LETTERS = ("A", "B", "C", "D", "E")
LETTER_IDS = [encode(" " + letter) for letter in LETTERS]
if any(len(ids) != 1 for ids in LETTER_IDS):
    raise RuntimeError("Answer letters must be single tokens for this tokenizer.")
LETTER_IDS = [ids[0] for ids in LETTER_IDS]

exam, exam_meta = [], []
for lang, df in load_eval_rows(EVAL_LANGUAGES, N_EVAL, SEED, reference_lang="en").items():
    for _, row in df.iterrows():
        ex = build_example(tokenizer, row)
        exam.append(ScoredSequence(ids=tuple(ex["ids"][0].tolist()), span=ex["passage_span"],
                                   choice_ids=tuple(LETTER_IDS[:4])))
        exam_meta.append({"task": "belebele", "lang": lang, "label": ex["label"]})


def kazmmlu_rows(language: str, n: int) -> pd.DataFrame:
    """A subject-balanced sample of KazMMLU test questions in one language."""
    tag = "kaz" if language == "kk" else "rus"
    frames = []
    for subset in datasets.get_dataset_config_names("MBZUAI/KazMMLU"):
        if subset.endswith(f"in {tag})"):
            frames.append(load_dataset("MBZUAI/KazMMLU", subset, split="test").to_pandas())
    pool = pd.concat(frames, ignore_index=True)
    per_subject = max(1, n // pool.Subset.nunique())
    parts = [group.sample(n=min(per_subject, len(group)), random_state=SEED) for _, group in pool.groupby("Subset")]
    sample = pd.concat(parts, ignore_index=True)
    return sample.sample(n=min(n, len(sample)), random_state=SEED).reset_index(drop=True)


for lang in ("kk", "ru"):
    for _, row in kazmmlu_rows(lang, KAZMMLU_PER_LANGUAGE).iterrows():
        options = [(L_, str(row[f"Option {L_}"])) for L_ in LETTERS if str(row[f"Option {L_}"]) not in ("None", "nan", "")]
        if row["Answer Key"] not in [o[0] for o in options]:
            continue
        text = f"Question: {row['Question'].strip()}" + "".join(f"\n{l}. {o}" for l, o in options) + "\nAnswer:"
        exam.append(ScoredSequence(ids=tuple(encode(text)), choice_ids=tuple(LETTER_IDS[LETTERS.index(l)] for l, _ in options)))
        exam_meta.append({"task": "kazmmlu", "lang": lang, "label": [o[0] for o in options].index(row["Answer Key"]),
                          "subject": row["Subset"]})

for probe_id, (lang, prompt, correct, wrong) in enumerate(FACT_PROBES):
    prompt_ids = encode(prompt)
    for rank, answer in enumerate([correct, *wrong]):
        ids = prompt_ids + encode(answer)
        exam.append(ScoredSequence(ids=tuple(ids), span=(len(prompt_ids), len(ids))))
        exam_meta.append({"task": "fact", "lang": lang, "probe": probe_id, "candidate": rank})
counts_by_task = pd.Series([m["task"] + "/" + m["lang"] for m in exam_meta]).value_counts().to_dict()
print(f"{len(exam)} exam sequences, {sum(len(s.ids) for s in exam):,} tokens: {counts_by_task}")


# %%
def wiki_sequences(config_name: str, docs: int, seed: int) -> list[ScoredSequence]:
    stream = load_dataset("wikimedia/wikipedia", config_name, split="train", streaming=True)
    out = []
    for row in stream.shuffle(seed=seed, buffer_size=10_000):
        text = row["text"].strip()
        if len(text) < 500:
            continue
        out.append(ScoredSequence(ids=tuple(tokenizer(text, truncation=True, max_length=CALIB_MAX_TOKENS).input_ids)))
        if len(out) == docs:
            return out
    raise RuntimeError(f"not enough articles in {config_name}")


calib_kkru = wiki_sequences("20231101.kk", CALIB_DOCS_PER_LANGUAGE, SEED + 1) + \
    wiki_sequences("20231101.ru", CALIB_DOCS_PER_LANGUAGE, SEED + 1)



# %% [markdown]
# ## 2. Calibration and the allocation plans

# %%
BASE_SEED = 20261118
TRAFFIC_COVERAGE = ((0.9,), (4, 3))


TEMPLATES = {}  # layer index -> bits chosen by the traffic plan; every control copies its 4-/3-bit counts


def plan(mode: str, seed: int = 0):
    """bits[experts] for one layer. All modes give every layer exactly the same number of 4-bit and 3-bit experts.
    The counts come from the traffic plan of the SAME layer (it runs first), because each configuration sees slightly
    different activations after its earlier layers are quantized, so its own traffic would give other counts."""
    def choose(config, index, layer, inputs, run_layer):
        traffic = router_traffic(layer, inputs, run_layer).astype(float)
        if mode == "traffic":
            TEMPLATES[index] = allocate_bits_by_coverage(traffic[None, :], *TRAFFIC_COVERAGE)
            return TEMPLATES[index][0]
        template = TEMPLATES[index]  # KeyError here means the traffic plan did not run first
        if mode == "random":
            score = np.random.default_rng([BASE_SEED, seed, index]).random((1, E))
        elif mode == "least-used first":
            score = -traffic[None, :]
        else:
            raise ValueError(mode)
        bits = allocate_like(template, score)[0]
        assert sorted(bits.tolist()) == sorted(template[0].tolist()), (mode, index)
        return bits
    return choose


MODES = {"traffic kk+ru": plan("traffic"), "random (seed 0)": plan("random", 0), "random (seed 1)": plan("random", 1), "random (seed 2)": plan("random", 2), "least-used first": plan("least-used first")}
PLANS = {name: {"method": "gptq", "bits": p} for name, p in MODES.items()}
CONFIGS = ["original", *MODES]

# %% [markdown]
# ## 3. Quantize and score (one streamed pass)

# %%
started = time.time()
log = lambda m: print(f"[{(time.time() - started) / 60:6.1f} min] {m}", flush=True)
prepare, gptq_stats = make_quantization_prepare(streamed, PLANS, group_size=GROUP_SIZE, attention_bits=ATTENTION_BITS, log=log)
scores = stream_evaluate(streamed, exam, CONFIGS, prepare=prepare, calibration={c: calib_kkru for c in MODES},
                         max_tokens=8192, layer_limit=LAYER_LIMIT, log=log)
log("done")

# %% [markdown]
# ## 4. Same memory? Then the results

# %%
layers_done = len(gptq_stats[next(iter(MODES))])
n4 = {c: [int((np.asarray(r["bits"]) == 4).sum()) for r in gptq_stats[c]] for c in MODES}
equal_memory = all(n4[c] == n4["traffic kk+ru"] for c in MODES)
print("4-bit experts per layer identical in all configurations (equal memory):", equal_memory)
assert equal_memory, "the controls must have exactly the same number of 4-bit experts per layer"
print("mean 4-bit experts per layer:", {c: round(float(np.mean(v)), 2) for c, v in n4.items()})

rows = []
for config in CONFIGS:
    for meta, s in zip(exam_meta, scores[config]):
        rows.append({"config": config, **meta, "logprob": s.span_logprob, "tokens": s.span_tokens,
                     "choice": int(np.argmax(s.choice_logits)) if s.choice_logits else -1})
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / "scores.csv", index=False)
bele = df[df.task == "belebele"]
ppl = bele.groupby(["config", "lang"]).apply(lambda g: float(np.exp(-g.logprob.sum() / g.tokens.sum())), include_groups=False).unstack("lang").reindex(CONFIGS)
growth = ppl / ppl.loc["original"] - 1
mmlu = df[df.task == "kazmmlu"].assign(ok=lambda d: d.choice == d.label).groupby(["config", "lang"]).ok.mean().unstack("lang").reindex(CONFIGS)
table = growth[["kk", "ru", "en"]].copy(); table["kazmmlu_kk"], table["kazmmlu_ru"] = mmlu["kk"], mmlu["ru"]
pd.set_option("display.width", 200)
print("Perplexity growth vs. the original (Belebele passages) and KazMMLU accuracy on the 762-question sample:")
print((table * 100).round(2))


def paired(a: str, b: str, lang: str, resamples: int = 10_000) -> dict:
    """Difference in log-probability per token (a minus b) over the same passages, with a paired bootstrap 95% interval."""
    x = bele[(bele.config == a) & (bele.lang == lang)].reset_index(drop=True)
    y = bele[(bele.config == b) & (bele.lang == lang)].reset_index(drop=True)
    d = (x.logprob.values - y.logprob.values); t = x.tokens.values
    rng = np.random.default_rng(0)
    idx = rng.integers(0, len(d), size=(resamples, len(d)))
    boots = d[idx].sum(1) / t[idx].sum(1)
    return {"nats_per_token": float(d.sum() / t.sum()), "ci95": [float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))]}


tests = {f"{a} minus {b} [{lang}]": paired(a, b, lang)
         for lang in ("kk", "ru") for a, b in (("traffic kk+ru", "random (seed 0)"), ("traffic kk+ru", "random (seed 1)"), ("traffic kk+ru", "random (seed 2)"), ("traffic kk+ru", "least-used first"), ("random (seed 0)", "least-used first"))}
for k, v in tests.items():
    print(f"{k}: {v['nats_per_token']:+.4f} nats/token, 95% CI [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]  (positive = first is better)")
summary = {"versions": VERSIONS, "layer_limit": LAYER_LIMIT, "layers": layers_done, "equal_memory": equal_memory,
           "table_percent": (table * 100).round(3).to_dict(), "paired_tests": tests, "minutes": (time.time() - started) / 60,
           "n4_per_layer": n4[next(iter(MODES))]}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, indent=1))
if LAYER_LIMIT:
    print("\n[SMOKE TEST: only the first layers were quantized; the numbers are meaningless]")
