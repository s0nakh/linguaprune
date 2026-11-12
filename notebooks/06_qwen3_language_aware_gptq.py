# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 06 · Qwen3-30B-A3B on free Kaggle GPUs: language-aware GPTQ for Kazakh + Russian
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# Qwen3-30B-A3B is a multilingual MoE model that actually knows Kazakh (48 layers × 128 experts, 8 active,
# 30.5B parameters, ~61 GB in 16-bit). It does not fit on 2×T4, so it is processed **one layer at a time**
# straight from the official Kaggle model files (attached as an Input — no download, no disk quota).
#
# Same method as notebook 05, but bits per expert are chosen **in this run** from live router traffic on
# Kazakh + Russian calibration text. The exam is held out from calibration:
# - **KazMMLU** — 400 Kazakh + 400 Russian questions across subjects (letter chosen by likelihood);
# - **Belebele** — perplexity and reading comprehension in kk / ru / en;
# - **facts** — "Astana vs Almaty", currency, Abai — scored by likelihood.
#
# **Target, unchanged:** ≥ 3.5× smaller than 16-bit with kk and ru perplexity growing by at most 2%,
# and no drop in KazMMLU accuracy beyond noise. Quantization is simulated; memory is analytic.
#
# **Run:** add Input → Models → `qwen-lm/qwen-3` → Transformers → `30b-a3b-instruct-2507`; GPU T4 x2,
# Internet On; Save Version → Save & Run All (Commit). Set `LAYER_LIMIT = 2` first for a timing dry run.
#
# ---
#
# 🇷🇺 **06 · Qwen3-30B-A3B на бесплатных видеокартах Kaggle: языковой GPTQ для казахского и русского**
#
# Qwen3-30B-A3B — многоязычная MoE-модель, которая действительно знает казахский (48 слоёв × 128 экспертов,
# 8 активных, 30,5 млрд параметров, ~61 ГБ в 16 битах). На 2×T4 она не помещается, поэтому обрабатывается
# **по одному слою за раз** прямо из официальных файлов модели на Kaggle (подключены как Input — без
# скачивания и без расхода диска).
#
# Метод как в ноутбуке 05, но биты каждому эксперту выбираются **в этом же прогоне** по живому трафику
# роутера на казахском и русском калибровочном тексте. Экзамен отделён от калибровки:
# - **KazMMLU** — 400 казахских и 400 русских вопросов по разным предметам (буква выбирается по вероятности);
# - **Belebele** — перплексия и понимание текста на kk / ru / en;
# - **факты** — «Астана против Алматы», валюта, Абай — по вероятности.
#
# **Цель та же:** в ≥ 3,5 раза меньше 16-битной модели, перплексия kk и ru растёт не больше чем на 2%,
# а точность KazMMLU не падает сверх шума. Квантование имитируется, память посчитана.
#
# **Запуск:** Add Input → Models → `qwen-lm/qwen-3` → Transformers → `30b-a3b-instruct-2507`; GPU T4 x2,
# Internet On; Save Version → Save & Run All (Commit). Сначала поставь `LAYER_LIMIT = 2` для пробного прогона.

# %%
import subprocess
import sys

LINGUAPRUNE_REF = "v0.4.0"  # pinned release of the linguaprune package
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       f"git+https://github.com/s0nakh/linguaprune@{LINGUAPRUNE_REF}"])

# %%
import glob
import json
import platform
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
LAYER_LIMIT = None  # set to 2 for a timing dry run (results are then meaningless)
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
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. On Kaggle, enable Settings → Accelerator → GPU T4 x2.")
torch.manual_seed(SEED)
VERSIONS = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "datasets": datasets.__version__,
            "linguaprune": linguaprune.__version__}
print(VERSIONS, f"| GPU x{torch.cuda.device_count()} | model {MODEL_DIR}")
print(f"linguaprune {linguaprune.__version__} from {linguaprune.__file__} (pinned ref {LINGUAPRUNE_REF})")

# %% [markdown]
# ## 1. Model skeleton and the exam
#
# ---
#
# 🇷🇺 **1. Каркас модели и экзамен**

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

# %% [markdown]
# ## 2. Calibration text and live bit allocation
#
# For each layer, the kk+ru configurations measure router traffic on their own Kazakh + Russian
# calibration text and give 4 bits to the experts carrying the first 80% (4/3) or 60%/90% (4/3/2) of
# that traffic. The English control gets exactly the same number of 4-bit and 3-bit experts per layer,
# ranked by English traffic and calibrated on English text — equal memory, only the language differs.
#
# ---
#
# 🇷🇺 **2. Калибровочный текст и распределение битов на лету**
#
# В каждом слое варианты kk+ru меряют трафик роутера на своём казахском и русском калибровочном тексте и
# дают 4 бита экспертам, через которых идут первые 80% трафика (4/3) или 60%/90% (4/3/2). Английский контроль
# получает ровно столько же 4- и 3-битных экспертов в каждом слое, выбранных по английскому трафику и
# откалиброванных на английском тексте: память одинаковая, отличается только язык.

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
calib_en = wiki_sequences("20231101.en", 2 * CALIB_DOCS_PER_LANGUAGE, SEED + 1)

LIVE_BITS: dict[tuple[str, int], np.ndarray] = {}


def coverage_bits(cuts, tiers):
    def choose(config, index, layer, inputs, run_layer):
        traffic = router_traffic(layer, inputs, run_layer).astype(float)
        bits = allocate_bits_by_coverage(traffic[None, :], cuts, tiers)[0]
        LIVE_BITS[(config, index)] = bits
        return bits
    return choose


def english_like(template_config):
    def choose(config, index, layer, inputs, run_layer):
        traffic = router_traffic(layer, inputs, run_layer).astype(float)
        bits = allocate_like(LIVE_BITS[(template_config, index)][None, :], traffic[None, :])[0]
        LIVE_BITS[(config, index)] = bits
        return bits
    return choose


# Notebook 05 (OLMoE): kk+ru 4/3 at 80% traffic kept Kazakh within +2.2% but Russian lost +4.4% —
# Kazakh concentrates on few experts and takes the 4-bit slots. The 90% variant protects Russian too.
PLANS = {  # order matters: the English control copies the bit counts of "gptq kkru 4/3 @90%" in the same layer
    "gptq 4-bit (kk+ru calib)": {"method": "gptq", "bits": np.full((L, E), 4)},
    "gptq kkru 4/3 @80%": {"method": "gptq", "bits": coverage_bits((0.8,), (4, 3))},
    "gptq kkru 4/3 @90%": {"method": "gptq", "bits": coverage_bits((0.9,), (4, 3))},
    "gptq en 4/3 @90% (control)": {"method": "gptq", "bits": english_like("gptq kkru 4/3 @90%")},
    "gptq kkru 4/3/2 @80/95%": {"method": "gptq", "bits": coverage_bits((0.8, 0.95), (4, 3, 2))},
}
CALIBRATION = {name: (calib_en if "(control)" in name else calib_kkru) for name in PLANS}
CONFIGS = ["original", *PLANS]

# %% [markdown]
# ## 3. Streamed run
#
# ---
#
# 🇷🇺 **3. Потоковый прогон**

# %%
started = time.time()
prepare, gptq_stats = make_quantization_prepare(streamed, PLANS, group_size=GROUP_SIZE, attention_bits=ATTENTION_BITS)
scores = stream_evaluate(streamed, exam, CONFIGS, prepare=prepare, calibration=CALIBRATION, max_tokens=8192,
                         layer_limit=LAYER_LIMIT,
                         log=lambda m: print(f"[{(time.time() - started) / 60:6.1f} min] {m}", flush=True))
print(f"Streamed run took {(time.time() - started) / 60:.1f} min")

# %% [markdown]
# ## 4. Memory and results vs. the target
#
# ---
#
# 🇷🇺 **4. Память и результаты против цели**

# %%
cfg = streamed.config
per_expert = 3 * cfg.hidden_size * cfg.moe_intermediate_size
attn_per_layer = sum(t.numel() for k, t in streamed.read_layer_state(0).items() if "self_attn" in k and k.endswith("proj.weight"))
index_meta = json.loads((Path(MODEL_DIR) / "model.safetensors.index.json").read_text()).get("metadata", {})
total_params = int(index_meta["total_size"]) // 2  # checkpoint is 16-bit: 2 bytes per parameter
other_params = total_params - L * (attn_per_layer + E * per_expert)
fp16_gb = total_params * 2 / 1e9


def memory_gb(config: str) -> float:
    if config == "original":
        return fp16_gb
    layers = [s["bits"] for s in gptq_stats[config]]
    expert_bits = sum((np.asarray(b) + 32 / GROUP_SIZE).sum() for b in layers) * per_expert
    expert_bits += (L - len(layers)) * E * 16 * per_expert  # layers not reached in a dry run stay 16-bit
    return (expert_bits / 8 + L * attn_per_layer * (ATTENTION_BITS + 32 / GROUP_SIZE) / 8 + other_params * 2) / 1e9


rows = []
for config in CONFIGS:
    for meta, s in zip(exam_meta, scores[config]):
        rows.append({"config": config, **meta, "logprob": s.span_logprob, "tokens": s.span_tokens,
                     "choice": int(np.argmax(s.choice_logits)) if s.choice_logits else -1})
df = pd.DataFrame(rows)
df.to_csv(OUTPUT_DIR / "scores.csv", index=False)

bele = df[df.task == "belebele"]
ppl = bele.groupby(["config", "lang"]).apply(lambda g: float(np.exp(-g.logprob.sum() / g.tokens.sum())),
                                             include_groups=False).unstack("lang").reindex(CONFIGS)
growth = ppl / ppl.loc["original"] - 1
accuracy = lambda d: d.assign(ok=d.choice == d.label).groupby(["config", "lang"]).ok.mean().unstack("lang").reindex(CONFIGS)
bele_acc, mmlu_acc = accuracy(bele), accuracy(df[df.task == "kazmmlu"])
facts = df[df.task == "fact"]
best = facts.loc[facts.groupby(["config", "probe"]).logprob.idxmax()]
fact_acc = best.assign(ok=best.candidate == 0).groupby(["config", "lang"]).ok.mean().unstack("lang").reindex(CONFIGS)

table = growth[["kk", "ru", "en"]].copy()
table["kazmmlu_kk"], table["kazmmlu_ru"] = mmlu_acc["kk"], mmlu_acc["ru"]
table["memory_gb"] = [memory_gb(c) for c in CONFIGS]
table["compression"] = fp16_gb / table["memory_gb"]
table["meets_target"] = ((table.compression >= TARGET_COMPRESSION) & (table.kk <= TARGET_MAX_PPL_GROWTH)
                         & (table.ru <= TARGET_MAX_PPL_GROWTH)).astype(bool)
pd.set_option("display.width", 220)
print(f"16-bit size {fp16_gb:.1f} GB. Belebele perplexity growth vs. original (0.02 = +2%), KazMMLU accuracy, memory:")
print(table.round(3))
print("\nBelebele accuracy:"); print(bele_acc.round(3))
print("\nFactual probes answered correctly:"); print(fact_acc.round(2))
print(f"\nKazMMLU questions: {int((df[(df.config == 'original') & (df.task == 'kazmmlu')]).shape[0])} "
      f"(95% CI of an accuracy near 0.5 is about ±{1.96 * (0.25 / KAZMMLU_PER_LANGUAGE) ** 0.5:.3f} per language)")
winners = table.index[table.meets_target].tolist()
print(f"\nTarget: {'MET by ' + ', '.join(winners) if winners else 'NOT met by any configuration'}"
      + ("  [DRY RUN — meaningless]" if LAYER_LIMIT else ""))

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
    "model": "Qwen/Qwen3-30B-A3B-Instruct-2507 (Kaggle qwen-lm/qwen-3)", "model_dir": MODEL_DIR, "versions": VERSIONS,
    "streamed": True, "compute_dtype": "float32", "layer_limit": LAYER_LIMIT, "fp16_gb": fp16_gb,
    "quantization": f"simulated; experts GPTQ group {GROUP_SIZE}; attention RTN {ATTENTION_BITS}-bit; embeddings 16-bit",
    "exam": counts_by_task, "target": {"compression": TARGET_COMPRESSION, "max_ppl_growth_kk_ru": TARGET_MAX_PPL_GROWTH},
    "table": table.reset_index().to_dict(orient="records"),
    "belebele_accuracy": bele_acc.reset_index().to_dict(orient="records"),
    "fact_accuracy": fact_acc.reset_index().to_dict(orient="records"),
    "bits_per_layer": {c: [s["bits"] for s in v] for c, v in gptq_stats.items()},
    "gptq_rtn_fallback_experts": {c: [s["rtn_fallback_experts"] for s in v] for c, v in gptq_stats.items()},
    "target_met_by": winners,
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(to_json(summary), ensure_ascii=False, indent=2, allow_nan=False))
print("Done:", sorted(p.name for p in OUTPUT_DIR.iterdir()))
