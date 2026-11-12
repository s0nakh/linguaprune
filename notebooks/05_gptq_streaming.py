# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 05 · GPTQ calibrated on Kazakh + Russian, run layer by layer
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# Notebook 04 showed that language-aware bit allocation beats English and random allocation, but plain
# round-to-nearest (RTN) quantization lost +6% even at 4 bits. Here the quantizer is **GPTQ**: it looks
# at real inputs of every expert and compensates each rounding error with the remaining weights.
# Calibration text is **Kazakh + Russian Wikipedia**; the evaluation is Belebele (never seen during
# calibration) plus factual questions scored by likelihood ("Astana" vs "Almaty").
#
# The model is processed **one layer at a time** (streaming). OLMoE would fit on the GPU anyway; this is
# the dress rehearsal for Qwen3-30B (61 GB), which only fits this way on free Kaggle GPUs.
#
# **Target, unchanged:** ≥ 3.5× smaller than fp16 with kk and ru perplexity growing by at most 2%.
# Attention is stored in 8 bits, embeddings and output layer stay fp16. Quantization is simulated;
# memory is computed for a real low-bit format. Speed is not measured.
#
# **Run:** GPU T4 x2, Internet On, Save Version → Save & Run All (Commit), ~1 hour.
#
# ---
#
# 🇷🇺 **05 · GPTQ с калибровкой на казахском и русском, по одному слою за раз**
#
# Ноутбук 04 показал, что распределение битов по языку лучше английского и случайного, но простое
# округление (RTN) теряло +6% даже на 4 битах. Здесь квантователь — **GPTQ**: он смотрит на настоящие
# входы каждого эксперта и компенсирует ошибку округления остальными весами. Калибровка — **казахская и
# русская Википедия**; экзамен — Belebele (его калибровка не видит) и фактические вопросы, оцениваемые по
# вероятности («Астана» против «Алматы»).
#
# Модель обрабатывается **по одному слою за раз** (потоково). OLMoE влезла бы и целиком; это генеральная
# репетиция для Qwen3-30B (61 ГБ), которая на бесплатных видеокартах Kaggle помещается только так.
#
# **Цель та же:** в ≥ 3,5 раза меньше fp16, а перплексия kk и ru растёт не больше чем на 2%.
# Внимание хранится в 8 битах, эмбеддинги и выходной слой — в fp16. Квантование имитируется,
# память посчитана для настоящего низкобитного формата. Скорость не измеряется.
#
# **Запуск:** GPU T4 x2, Internet On, Save Version → Save & Run All (Commit), ~1 час.

# %%
import subprocess
import sys

LINGUAPRUNE_REF = "v0.3.1"  # pinned release of the linguaprune package
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       f"git+https://github.com/s0nakh/linguaprune@{LINGUAPRUNE_REF}"])

# %%
import json
import platform
import time
import urllib.request
from pathlib import Path

import datasets
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import snapshot_download
from transformers import AutoTokenizer

import linguaprune
from linguaprune import allocate_bits_by_coverage, allocate_like, language_frequency_score, row_share
from linguaprune.belebele import answer_token_ids, build_example, load_eval_rows
from linguaprune.gptq import make_quantization_prepare
from linguaprune.streaming import ScoredSequence, StreamedModel, stream_evaluate

MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"  # same weights as notebooks 01–04
REPO_RAW = f"https://raw.githubusercontent.com/s0nakh/linguaprune/{LINGUAPRUNE_REF}"
ROUTING_URL = f"{REPO_RAW}/experiments/02/calibration.npz"  # router traffic per expert, from notebook 02
EVAL_LANGUAGES = {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}
N_EVAL = 150
SEED = 0
GROUP_SIZE = 128
ATTENTION_BITS = 8
CALIB_DOCS_PER_LANGUAGE = 64  # Wikipedia articles per calibration language
CALIB_MAX_TOKENS = 512
TARGET_COMPRESSION = 3.5
TARGET_MAX_PPL_GROWTH = 0.02
LAYER_LIMIT = None  # set to 2 for a timing dry run (results are then meaningless)
# Factual probes: prompt, correct continuation, wrong continuations (scored by summed log-probability)
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

KAGGLE_WORKING = Path("/kaggle/working")
WORK_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")
OUTPUT_DIR = WORK_DIR / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. On Kaggle, enable Settings → Accelerator → GPU T4 x2.")
torch.manual_seed(SEED)
VERSIONS = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "datasets": datasets.__version__,
            "linguaprune": linguaprune.__version__}
print(VERSIONS, f"| GPU x{torch.cuda.device_count()}")
print(f"linguaprune {linguaprune.__version__} from {linguaprune.__file__} (pinned ref {LINGUAPRUNE_REF})")

# %% [markdown]
# ## 1. Weights, tokenizer and the exam
#
# ---
#
# 🇷🇺 **1. Веса, токенизатор и экзамен**

# %%
model_dir = snapshot_download(MODEL_ID, revision=MODEL_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt"])
tokenizer = AutoTokenizer.from_pretrained(model_dir)
streamed = StreamedModel(model_dir, device="cuda:0", dtype=torch.float32)  # fp32 per layer: no fp16 overflow
L, E = streamed.num_layers, streamed.num_experts
print(f"Layers {L}, experts per layer {E}")

ANSWER_IDS = tuple(answer_token_ids(tokenizer))
EVAL_ROWS = load_eval_rows(EVAL_LANGUAGES, N_EVAL, SEED, reference_lang="en")
exam, exam_meta = [], []
for lang, df in EVAL_ROWS.items():
    for _, row in df.iterrows():
        ex = build_example(tokenizer, row)
        exam.append(ScoredSequence(ids=tuple(ex["ids"][0].tolist()), span=ex["passage_span"], choice_ids=ANSWER_IDS))
        exam_meta.append({"task": "belebele", "lang": lang, "label": ex["label"]})


def encode(text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


for probe_id, (lang, prompt, correct, wrong) in enumerate(FACT_PROBES):
    prompt_ids = encode(prompt)
    for rank, answer in enumerate([correct, *wrong]):
        ids = prompt_ids + encode(answer)
        exam.append(ScoredSequence(ids=tuple(ids), span=(len(prompt_ids), len(ids))))
        exam_meta.append({"task": "fact", "lang": lang, "probe": probe_id, "candidate": rank})
print(f"{len(exam)} exam sequences, {sum(len(s.ids) for s in exam):,} tokens")

# %% [markdown]
# ## 2. Calibration text and bit plans
#
# GPTQ calibration: Kazakh + Russian Wikipedia for our configurations, English Wikipedia for the
# English control. Bit allocation reuses notebook 02's router traffic, exactly as in notebook 04.
#
# ---
#
# 🇷🇺 **2. Калибровочные тексты и планы битов**
#
# Калибровка GPTQ: казахская и русская Википедия для наших вариантов, английская — для контроля.
# Распределение битов берёт трафик роутера из ноутбука 02, точно как в ноутбуке 04.

# %%
def wiki_sequences(config_name: str, docs: int, seed: int) -> list[ScoredSequence]:
    stream = load_dataset("wikimedia/wikipedia", config_name, split="train", streaming=True)
    out = []
    for row in stream.shuffle(seed=seed, buffer_size=10_000):
        text = row["text"].strip()
        if len(text) < 500:
            continue
        ids = tokenizer(text, truncation=True, max_length=CALIB_MAX_TOKENS).input_ids
        out.append(ScoredSequence(ids=tuple(ids)))
        if len(out) == docs:
            return out
    raise RuntimeError(f"not enough articles in {config_name}")


CALIB_SEED = SEED + 1  # different articles from notebook 02's router statistics
calib_kkru = wiki_sequences("20231101.kk", CALIB_DOCS_PER_LANGUAGE, CALIB_SEED) + \
    wiki_sequences("20231101.ru", CALIB_DOCS_PER_LANGUAGE, CALIB_SEED)
calib_en = wiki_sequences("20231101.en", 2 * CALIB_DOCS_PER_LANGUAGE, CALIB_SEED)

urllib.request.urlretrieve(ROUTING_URL, WORK_DIR / "nb02_routing.npz")
routing = np.load(WORK_DIR / "nb02_routing.npz")
counts = {lang: routing[f"{lang}_counts"] for lang in ("kk", "ru", "en")}
if counts["kk"].shape != (L, E):
    raise RuntimeError(f"routing statistics {counts['kk'].shape} do not match the model {(L, E)}")
kkru_score = language_frequency_score(counts, ("kk", "ru"))
bits_43 = allocate_bits_by_coverage(kkru_score, (0.8,), (4, 3))
bits_432 = allocate_bits_by_coverage(kkru_score, (0.6, 0.9), (4, 3, 2))
PLANS = {
    "rtn 4-bit": {"method": "rtn", "bits": np.full((L, E), 4)},
    "gptq 4-bit (kk+ru calib)": {"method": "gptq", "bits": np.full((L, E), 4)},
    "gptq kkru 4/3": {"method": "gptq", "bits": bits_43},
    "gptq en 4/3 (control)": {"method": "gptq", "bits": allocate_like(bits_43, row_share(counts["en"]))},
    "gptq kkru 4/3/2": {"method": "gptq", "bits": bits_432},
}
CALIBRATION = {name: (calib_en if "(control)" in name else calib_kkru) for name, p in PLANS.items() if p["method"] == "gptq"}
CONFIGS = ["original", *PLANS]

experts_params = L * E * (streamed.config.intermediate_size * 2 * streamed.config.hidden_size
                          + streamed.config.hidden_size * streamed.config.intermediate_size)
attn_params = sum(t.numel() for k, t in streamed.read_layer_state(0).items() if "self_attn" in k and "proj" in k) * L
total_params = 6_919_161_856
other_params = total_params - experts_params - attn_params


def memory_gb(bits) -> float:
    if bits is None:
        return total_params * 2 / 1e9
    per_expert = experts_params / (L * E)
    expert_bits = (np.asarray(bits) + 32 / GROUP_SIZE) * per_expert
    return (expert_bits.sum() / 8 + attn_params * (ATTENTION_BITS + 32 / GROUP_SIZE) / 8 + other_params * 2) / 1e9


for name in CONFIGS:
    gb = memory_gb(PLANS[name]["bits"] if name in PLANS else None)
    print(f"{name:>26}: {gb:5.2f} GB, {memory_gb(None) / gb:4.2f}× smaller than fp16")

# %% [markdown]
# ## 3. Streamed run: every configuration, layer by layer
#
# ---
#
# 🇷🇺 **3. Потоковый прогон: все варианты, слой за слоем**

# %%
started = time.time()
prepare, gptq_stats = make_quantization_prepare(streamed, PLANS, group_size=GROUP_SIZE, attention_bits=ATTENTION_BITS)
scores = stream_evaluate(streamed, exam, CONFIGS, prepare=prepare, calibration=CALIBRATION, max_tokens=8192,
                         layer_limit=LAYER_LIMIT,
                         log=lambda m: print(f"[{(time.time() - started) / 60:5.1f} min] {m}", flush=True))
print(f"Streamed run took {(time.time() - started) / 60:.1f} min")

# %% [markdown]
# ## 4. Results vs. the target
#
# ---
#
# 🇷🇺 **4. Результаты против цели**

# %%
rows = []
for config in CONFIGS:
    for meta, s in zip(exam_meta, scores[config]):
        rows.append({"config": config, **meta, "logprob": s.span_logprob, "tokens": s.span_tokens,
                     "choice": int(np.argmax(s.choice_logits)) if s.choice_logits else -1})
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / "scores.csv", index=False)

bele = df[df.task == "belebele"]
ppl = bele.groupby(["config", "lang"]).apply(lambda g: float(np.exp(-g.logprob.sum() / g.tokens.sum())), include_groups=False)
ppl = ppl.unstack("lang").reindex(CONFIGS)
growth = ppl / ppl.loc["original"] - 1
acc = bele.assign(ok=bele.choice == bele.label).groupby(["config", "lang"]).ok.mean().unstack("lang").reindex(CONFIGS)

facts = df[df.task == "fact"].copy()
facts["norm"] = facts.logprob  # summed log-probability of the answer tokens
best = facts.loc[facts.groupby(["config", "probe"]).norm.idxmax()]
fact_acc = best.assign(ok=best.candidate == 0).groupby(["config", "lang"]).ok.mean().unstack("lang").reindex(CONFIGS)

table = growth[["kk", "ru", "en"]].copy()
table["memory_gb"] = [memory_gb(PLANS[c]["bits"] if c in PLANS else None) for c in CONFIGS]
table["compression"] = memory_gb(None) / table["memory_gb"]
table["meets_target"] = ((table.compression >= TARGET_COMPRESSION) & (table.kk <= TARGET_MAX_PPL_GROWTH)
                         & (table.ru <= TARGET_MAX_PPL_GROWTH)).astype(bool)
pd.set_option("display.width", 200)
print("Perplexity growth vs. original (0.02 = +2%):")
print(table.round(3))
print("\nBelebele accuracy (chance 0.25):")
print(acc.round(3))
print("\nFactual probes answered correctly (share, 3 per language):")
print(fact_acc.round(2))
winners = table.index[table.meets_target].tolist()
print(f"\nTarget: {'MET by ' + ', '.join(winners) if winners else 'NOT met by any configuration'}")

# %%
def to_json(value):
    if isinstance(value, dict):
        return {str(k): to_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_json(v) for v in value]
    if isinstance(value, (np.bool_, bool)):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        return None if np.isnan(value) else float(value)
    return value


summary = {
    "model": MODEL_ID, "model_revision": MODEL_REVISION, "versions": VERSIONS,
    "streamed": True, "compute_dtype": "float32", "layer_limit": LAYER_LIMIT,
    "quantization": f"simulated; experts GPTQ or RTN, group {GROUP_SIZE}; attention RTN {ATTENTION_BITS}-bit; embeddings fp16",
    "calibration": {"kk+ru docs": len(calib_kkru), "en docs": len(calib_en), "max_tokens": CALIB_MAX_TOKENS},
    "target": {"compression": TARGET_COMPRESSION, "max_ppl_growth_kk_ru": TARGET_MAX_PPL_GROWTH},
    "table": table.reset_index().to_dict(orient="records"),
    "belebele_accuracy": acc.reset_index().to_dict(orient="records"),
    "fact_accuracy": fact_acc.reset_index().to_dict(orient="records"),
    "gptq_rtn_fallback_experts": {c: [s["rtn_fallback_experts"] for s in v] for c, v in gptq_stats.items()},
    "target_met_by": winners,
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(to_json(summary), ensure_ascii=False, indent=2, allow_nan=False))
print("Done:", sorted(p.name for p in OUTPUT_DIR.iterdir()))
