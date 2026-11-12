# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 07 · A real low-bit file: Qwen3-30B-A3B compressed for Kazakh + Russian, verified from disk
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# Notebook 06 met the target with *simulated* quantization and *computed* memory. This notebook makes it real:
# 1. quantize Qwen3-30B-A3B layer by layer exactly as in 06 (GPTQ calibrated on kk+ru Wikipedia);
# 2. **write each layer to disk** in the standard GPTQ checkpoint layout (4-bit or 3-bit codes, fp16 scales);
# 3. **read the written files back** and run the whole exam again from disk, so every number comes from the file;
# 4. measure the real size on disk;
# 5. upload the verified models to Hugging Face as **private** repositories (the author publishes them after review).
#
# Two models are written: **kk+ru 4/3-bit @90%** (the result of 06) and **kk+ru uniform 4-bit** (one width
# everywhere, for runtimes that cannot mix widths inside a layer).
#
# **Run:** Input → Models → `qwen-lm/qwen-3` → Transformers → `30b-a3b-instruct-2507`; Add-ons → Secrets → `HF_TOKEN`
# (write access); GPU T4, Internet On. Set `LAYER_LIMIT = 2` for a dry run (no upload).
#
# ---
#
# 🇷🇺 **07 · Настоящий файл: Qwen3-30B-A3B, сжатая для казахского и русского, с проверкой прямо с диска**
#
# Ноутбук 06 достиг цели при *имитации* квантования и *посчитанной* памяти. Здесь всё по-настоящему:
# 1. квантуем Qwen3-30B-A3B по одному слою точно как в 06 (GPTQ с калибровкой на Википедии kk+ru);
# 2. **записываем каждый слой на диск** в стандартном формате GPTQ (коды по 4 или 3 бита, масштабы в fp16);
# 3. **читаем записанные файлы обратно** и заново прогоняем весь экзамен с диска: каждая цифра — из файла;
# 4. меряем настоящий размер на диске;
# 5. загружаем проверенные модели на Hugging Face **приватно** (автор публикует их после просмотра).
#
# Пишутся две модели: **kk+ru 4/3 бита @90%** (результат 06) и **kk+ru равномерно 4 бита** (одна разрядность
# везде — для программ, которые не умеют смешивать разрядности внутри слоя).
#
# **Запуск:** Input → Models → `qwen-lm/qwen-3` → Transformers → `30b-a3b-instruct-2507`; Add-ons → Secrets →
# `HF_TOKEN` (с правом записи); GPU T4, Internet On. Для пробного прогона поставь `LAYER_LIMIT = 2` (без загрузки).

# %%
import subprocess
import sys

LINGUAPRUNE_REF = "83d860b3cb50d8249f08ce6b9c5f382624b3b9aa"  # pinned commit of the linguaprune package (v0.5.0)
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       f"git+https://github.com/s0nakh/linguaprune@{LINGUAPRUNE_REF}"])

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
from linguaprune.export import GPTQCheckpointWriter
from linguaprune.gptq import make_quantization_prepare, router_traffic
from linguaprune.quant import allocate_bits_by_coverage
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
LAYER_LIMIT = None  # set to 2 for a dry run: two layers are written and verified, nothing is uploaded
HF_USER = "s0nakh"
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
MIN_FREE_DISK_GB = 40
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
if LAYER_LIMIT is None and free_gb < MIN_FREE_DISK_GB:
    raise RuntimeError(f"Only {free_gb:.0f} GB free in {CHECKPOINT_ROOT}; two checkpoints need ~{MIN_FREE_DISK_GB} GB.")
torch.manual_seed(SEED)
VERSIONS = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "datasets": datasets.__version__,
            "linguaprune": linguaprune.__version__}
print(VERSIONS, f"| GPU x{torch.cuda.device_count()} | model {MODEL_DIR} | {free_gb:.0f} GB free disk")


def hf_token() -> str | None:
    """The HF_TOKEN Kaggle secret, or None. The value is never printed."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as error:  # secret not attached, or not running on Kaggle
        print(f"HF_TOKEN secret not available ({type(error).__name__}); models will not be uploaded.")
        return None


TOKEN = hf_token()
if LAYER_LIMIT is None and not TOKEN:  # the written files live only in this session: without upload they are lost
    raise RuntimeError("Full run needs the HF_TOKEN secret: Add-ons → Secrets → enable HF_TOKEN, then run again.")
if TOKEN:
    from huggingface_hub import HfApi
    whoami = HfApi(token=TOKEN).whoami()["name"]
    if whoami != HF_USER:
        raise RuntimeError(f"HF_TOKEN belongs to {whoami!r}, expected {HF_USER!r}")
    print(f"Hugging Face token OK for {whoami}")

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

# %% [markdown]
# ## 2. Calibration and the two models to write
#
# ---
#
# 🇷🇺 **2. Калибровка и две модели для записи**

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


def coverage_bits(cuts, tiers):
    def choose(config, index, layer, inputs, run_layer):
        traffic = router_traffic(layer, inputs, run_layer).astype(float)
        return allocate_bits_by_coverage(traffic[None, :], cuts, tiers)[0]
    return choose


MODELS = {  # config name -> (Hugging Face repo name, bit plan)
    "kk+ru 4/3-bit @90%": ("Qwen3-30B-A3B-Instruct-2507-kk-ru-GPTQ-4-3bit", coverage_bits((0.9,), (4, 3))),
    "kk+ru 4-bit": ("Qwen3-30B-A3B-Instruct-2507-kk-ru-GPTQ-4bit", np.full((L, E), 4)),
}
PLANS = {name: {"method": "gptq", "bits": plan} for name, (_, plan) in MODELS.items()}
CONFIGS = ["original", *MODELS]
WRITERS = {
    name: GPTQCheckpointWriter(
        CHECKPOINT_ROOT / repo, streamed.reader, group_size=GROUP_SIZE, default_bits=4, attention_bits=ATTENTION_BITS,
        description={"base_model": BASE_MODEL, "configuration": name, "calibration": "Wikipedia kk + ru, 64 + 64 articles",
                     "linguaprune": LINGUAPRUNE_REF})
    for name, (repo, _) in MODELS.items()
}

# %% [markdown]
# ## 3. Quantize, write to disk, and score (one streamed pass)
#
# ---
#
# 🇷🇺 **3. Квантование, запись на диск и оценка (один потоковый проход)**

# %%
started = time.time()
log = lambda m: print(f"[{(time.time() - started) / 60:6.1f} min] {m}", flush=True)
prepare, gptq_stats = make_quantization_prepare(
    streamed, PLANS, group_size=GROUP_SIZE, attention_bits=ATTENTION_BITS, log=log,
    on_quantized=lambda config, index, layer, record: WRITERS[config].write_layer(config, index, layer, record))
simulated = stream_evaluate(streamed, exam, CONFIGS, prepare=prepare, calibration={c: calib_kkru for c in MODELS},
                            max_tokens=8192, layer_limit=LAYER_LIMIT, log=log)
written_layers = L if LAYER_LIMIT is None else min(LAYER_LIMIT, L)
summaries = {name: writer.finish(num_layers=written_layers) for name, writer in WRITERS.items()}
log("quantized and written")

# %% [markdown]
# ## 4. Read the files back and run the exam from disk
#
# Every score below comes from the written checkpoint, dequantized by the same reader. It must equal the
# score of the in-memory model from step 3; any difference means the file is not the model we measured.
#
# ---
#
# 🇷🇺 **4. Читаем файлы обратно и сдаём экзамен с диска**
#
# Каждая оценка ниже получена из записанного файла. Она должна совпасть с оценкой модели в памяти из шага 3;
# любое расхождение значит, что файл — не та модель, которую мы измерили.

# %%
from_file, max_diff = {}, {}
for name, (repo, _) in MODELS.items():
    reloaded = StreamedModel(CHECKPOINT_ROOT / repo, device="cuda:0", dtype=torch.float32)
    from_file[name] = stream_evaluate(reloaded, exam, ["file"], max_tokens=8192, layer_limit=LAYER_LIMIT, log=log)["file"]
    diffs = [abs(a.span_logprob - b.span_logprob) for a, b in zip(simulated[name], from_file[name])]
    diffs += [abs(x - y) for a, b in zip(simulated[name], from_file[name]) for x, y in zip(a.choice_logits, b.choice_logits)]
    max_diff[name] = max(diffs)
    print(f"{name}: largest score difference simulated vs. from file = {max_diff[name]:.2e}")
    del reloaded
VERIFIED = all(d <= 1e-3 for d in max_diff.values())  # identical up to GPU summation order
print("Files reproduce the measured models:", VERIFIED)
if not VERIFIED:
    raise RuntimeError("A written checkpoint does not reproduce its scores; nothing will be uploaded.")


def disk_gb(path: Path) -> float:
    return sum(p.stat().st_size for p in path.rglob("*.safetensors")) / 1e9


index_meta = json.loads((Path(MODEL_DIR) / "model.safetensors.index.json").read_text()).get("metadata", {})
fp16_gb = int(index_meta["total_size"]) / 1e9
size_gb = {name: disk_gb(CHECKPOINT_ROOT / repo) for name, (repo, _) in MODELS.items()}

# %% [markdown]
# ## 5. Results from the files
#
# ---
#
# 🇷🇺 **5. Результаты с файлов**

# %%
scores = {"original": simulated["original"], **from_file}
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
table["disk_gb"] = [fp16_gb if c == "original" else size_gb[c] for c in CONFIGS]
table["compression"] = fp16_gb / table["disk_gb"]
table["meets_target"] = ((table.compression >= TARGET_COMPRESSION) & (table.kk <= TARGET_MAX_PPL_GROWTH)
                         & (table.ru <= TARGET_MAX_PPL_GROWTH)).astype(bool)
pd.set_option("display.width", 220)
print(f"Original {fp16_gb:.1f} GB on disk. Every quantized row is scored from its written file:")
print(table.round(3))
print("\nBelebele accuracy:"); print(bele_acc.round(3))
print("\nFactual probes answered correctly:"); print(fact_acc.round(2))
if LAYER_LIMIT:
    print("\n[DRY RUN — only the first layers were written; sizes and scores are meaningless]")

# %% [markdown]
# ## 6. Upload to Hugging Face (private)
#
# Only after the files were verified, and never in a dry run. Repositories are created **private**; the author
# reviews the model card and makes them public by hand.
#
# ---
#
# 🇷🇺 **6. Загрузка на Hugging Face (приватно)**
#
# Только после проверки файлов и никогда в пробном прогоне. Репозитории создаются **приватными**; автор
# просматривает карточку модели и открывает их вручную.

# %%
def pct(x: float) -> str:
    return f"{100 * x:+.1f}%"


def model_card(name: str, repo: str) -> str:
    row, base = table.loc[name], table.loc["original"]
    mixed = "4/3" in name
    return f"""---
license: apache-2.0
base_model: {BASE_MODEL}
language: [kk, ru, en]
tags: [gptq, moe, quantized, kazakh, russian, linguaprune]
---

# {repo}

[{BASE_MODEL}](https://huggingface.co/{BASE_MODEL}) compressed with **language-aware GPTQ** for Kazakh and Russian
by [LinguaPrune](https://github.com/s0nakh/linguaprune). Every expert is GPTQ-quantized with calibration on Kazakh +
Russian Wikipedia; {"experts that carry 90% of kk+ru router traffic keep 4 bits, the rest get 3 bits" if mixed else "all experts use 4 bits"}.
Attention is 8-bit, embeddings and norms are unchanged.

| | Original | This model |
|---|---|---|
| Size on disk | {fp16_gb:.1f} GB | **{row.disk_gb:.1f} GB ({row.compression:.1f}× smaller)** |
| Kazakh perplexity (Belebele passages) | — | {pct(row.kk)} |
| Russian perplexity | — | {pct(row.ru)} |
| KazMMLU, Kazakh | {100 * base.kazmmlu_kk:.1f}% | {100 * row.kazmmlu_kk:.1f}% |
| KazMMLU, Russian | {100 * base.kazmmlu_ru:.1f}% | {100 * row.kazmmlu_ru:.1f}% |

All numbers are measured **from these files** ({int((df.task == "kazmmlu").sum() / len(CONFIGS))} KazMMLU questions,
150 Belebele passages per language). Method, controls and limitations: [research log](https://github.com/s0nakh/linguaprune/blob/main/docs/research-log.md).

## Format

Standard GPTQ checkpoint (`qweight` / `qzeros` / `scales` / `g_idx`, group size {GROUP_SIZE}, asymmetric, no act-order).
{"Experts of the same layer have different bit widths, declared in `quantization_config.dynamic`. Most runtimes cannot run mixed widths inside one MoE layer yet; use the LinguaPrune reader below." if mixed else "8-bit attention is declared in `quantization_config.dynamic`."}
Compatibility with vLLM and other runtimes is **not tested yet**.

```bash
pip install "git+https://github.com/s0nakh/linguaprune@{LINGUAPRUNE_REF}"
```

```python
from huggingface_hub import snapshot_download
from linguaprune.streaming import StreamedModel  # layer-by-layer evaluation, fits a small GPU

model = StreamedModel(snapshot_download("{HF_USER}/{repo}"), device="cuda")
```

## Limitations

One model, one seed. Kazakh KazMMLU trends 2–3 points lower than the original in notebook 06 (not significant on
~370 questions). The license of the base model (Apache-2.0) applies.
"""


if LAYER_LIMIT or not TOKEN:
    print("Upload skipped:", "dry run" if LAYER_LIMIT else "no HF_TOKEN")
else:
    api = HfApi(token=TOKEN)
    for name, (repo, _) in MODELS.items():
        folder = CHECKPOINT_ROOT / repo
        (folder / "README.md").write_text(model_card(name, repo))
        repo_id = f"{HF_USER}/{repo}"
        api.create_repo(repo_id, private=True, exist_ok=True)
        api.upload_large_folder(repo_id=repo_id, folder_path=folder, repo_type="model")
        log(f"uploaded {repo_id} (private)")

# %%
summary = {
    "model": BASE_MODEL, "versions": VERSIONS, "layer_limit": LAYER_LIMIT, "fp16_gb": fp16_gb,
    "quantization": f"real GPTQ checkpoint; experts group {GROUP_SIZE}; attention {ATTENTION_BITS}-bit; scored from file",
    "exam": counts_by_task, "verified_max_score_difference": max_diff, "disk_gb": size_gb,
    "table": json.loads(table.reset_index().to_json(orient="records")),
    "belebele_accuracy": json.loads(bele_acc.reset_index().to_json(orient="records")),
    "fact_accuracy": json.loads(fact_acc.reset_index().to_json(orient="records")),
    "checkpoints": summaries,
    "uploaded_private": [f"{HF_USER}/{repo}" for repo, _ in MODELS.values()] if (TOKEN and not LAYER_LIMIT) else [],
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("Done:", sorted(p.name for p in OUTPUT_DIR.iterdir()))
