# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 09 · How do popular compressed versions of Qwen3-30B-A3B treat Kazakh?
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# The most downloaded low-bit builds of Qwen3-30B-A3B-Instruct-2507 are GGUF files for llama.cpp (Ollama, LM Studio).
# Their quantizers are tuned and checked mostly on English. Here every build takes **exactly the same exam as
# notebooks 06–07** (same token ids): Belebele perplexity and accuracy in kk/ru/en, KazMMLU in kk/ru, factual probes.
#
# - Reference: the original 16-bit model, scored in notebook 07 (`experiments/07/full/scores.csv`).
# - Engine check: `Q6_K` is near-lossless; its numbers show how much llama.cpp itself differs from PyTorch.
# - LinguaPrune's own checkpoints are not re-run: their numbers come from notebook 07, measured from the files.
#
# **Run:** Accelerator GPU T4 x2, Internet On. No secrets needed (all models are public). Set `DRY_RUN = True` to
# check the pipeline on one small model and a few questions.
#
# ---
#
# 🇷🇺 **09 · Как популярные сжатые версии Qwen3-30B-A3B обходятся с казахским?**
#
# Самые скачиваемые сжатые версии Qwen3-30B-A3B-Instruct-2507 — файлы GGUF для llama.cpp (Ollama, LM Studio). Их
# настраивают и проверяют в основном на английском. Здесь каждая версия сдаёт **ровно тот же экзамен, что в
# ноутбуках 06–07** (те же токены): перплексия и точность Belebele на kk/ru/en, KazMMLU на kk/ru, фактические вопросы.
#
# - Точка отсчёта: исходная 16-битная модель из ноутбука 07.
# - Проверка движка: `Q6_K` почти без потерь; по ней видно, насколько сам llama.cpp отличается от PyTorch.
# - Модели LinguaPrune заново не запускаются: их цифры — из ноутбука 07, измерены с файлов.
#
# **Запуск:** GPU T4 x2, Internet On. Секреты не нужны (все модели открытые). `DRY_RUN = True` — проверка на одной
# маленькой модели и нескольких вопросах.

# %%
import os
import subprocess
import sys
import time

LINGUAPRUNE_REF = "d54eecfc6c8215a9ef69f752e2b79d521c572e3f"  # pinned commit of the linguaprune package (v0.5.1)
LLAMA_CPP_PYTHON_VERSION = "0.3.36"
pip = [sys.executable, "-m", "pip", "install", "-q"]
subprocess.check_call(pip + [f"git+https://github.com/s0nakh/linguaprune@{LINGUAPRUNE_REF}"])
started = time.time()
# llama.cpp with CUDA, compiled for the T4 only (sm_75) to keep the build short
import glob
import shutil

nvcc = shutil.which("nvcc") or next(iter(sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))), None)
print("nvcc:", nvcc, "| cmake:", shutil.which("cmake"), "| gcc:", shutil.which("gcc"))
if nvcc is None:
    raise RuntimeError("No CUDA compiler (nvcc) found on this machine; llama.cpp cannot be built with CUDA.")
build_env = {**os.environ, "CMAKE_ARGS": "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_NO_VMM=ON",  # no VMM: no libcuda link at build time
             "FORCE_CMAKE": "1",
             "CUDACXX": nvcc, "PATH": f"{os.path.dirname(nvcc)}:{os.environ['PATH']}",
             "CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count() or 4)}
if shutil.which("cmake") is None:
    subprocess.check_call(pip + ["cmake", "ninja"])
# the wheel is kept in the notebook output, so later notebooks can install it without rebuilding (~30 min)
WHEEL_DIR = "/kaggle/working/wheels" if os.path.isdir("/kaggle/working") else "wheels"
build = subprocess.run([sys.executable, "-m", "pip", "wheel", "-v", "--no-cache-dir", "--no-deps", "-w", WHEEL_DIR,
                        f"llama-cpp-python=={LLAMA_CPP_PYTHON_VERSION}"], env=build_env, capture_output=True, text=True)
if build.returncode != 0:
    print("\n".join((build.stdout + build.stderr).splitlines()[-80:]))
    raise RuntimeError("llama-cpp-python build failed; the last lines of the build log are above.")
subprocess.check_call(pip + [*glob.glob(f"{WHEEL_DIR}/llama_cpp_python-*.whl")])
print(f"llama-cpp-python built in {(time.time() - started) / 60:.1f} min")

# %%
import json
import platform
import traceback
import urllib.request
from pathlib import Path

import datasets
import llama_cpp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from llama_cpp import Llama
from transformers import AutoTokenizer

import linguaprune
from linguaprune.belebele import build_example, load_eval_rows

DRY_RUN = False  # True: one small model, a few questions per task
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SCORES_07 = "https://raw.githubusercontent.com/s0nakh/linguaprune/main/experiments/07/full/scores.csv"
SUMMARY_07 = "https://raw.githubusercontent.com/s0nakh/linguaprune/main/experiments/07/full/summary.json"
UNSLOTH = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
BARTOWSKI = "bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF"
COMPETITORS = [  # (label, repository, file), smallest last
    ("Q6_K (engine check)", UNSLOTH, "Qwen3-30B-A3B-Instruct-2507-Q6_K.gguf"),
    ("Q4_K_M", UNSLOTH, "Qwen3-30B-A3B-Instruct-2507-Q4_K_M.gguf"),
    ("UD-Q4_K_XL", UNSLOTH, "Qwen3-30B-A3B-Instruct-2507-UD-Q4_K_XL.gguf"),
    ("IQ4_XS", UNSLOTH, "Qwen3-30B-A3B-Instruct-2507-IQ4_XS.gguf"),
    ("UD-Q3_K_XL", UNSLOTH, "Qwen3-30B-A3B-Instruct-2507-UD-Q3_K_XL.gguf"),
    ("UD-IQ3_XXS", UNSLOTH, "Qwen3-30B-A3B-Instruct-2507-UD-IQ3_XXS.gguf"),
    ("UD-Q2_K_XL", UNSLOTH, "Qwen3-30B-A3B-Instruct-2507-UD-Q2_K_XL.gguf"),
    ("UD-IQ2_M", UNSLOTH, "Qwen3-30B-A3B-Instruct-2507-UD-IQ2_M.gguf"),
    ("bartowski IQ2_M", BARTOWSKI, "Qwen_Qwen3-30B-A3B-Instruct-2507-IQ2_M.gguf"),
]
if DRY_RUN:
    COMPETITORS = [c for c in COMPETITORS if c[0] == "bartowski IQ2_M"]
EVAL_LANGUAGES = {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}
N_EVAL, SEED, KAZMMLU_PER_LANGUAGE = 150, 0, 400  # identical to notebook 07
N_CTX = 4096
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
OUTPUT_DIR = (KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")) / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_ROOT = Path("/kaggle/tmp") if KAGGLE_WORKING.exists() else Path("downloads")
DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. On Kaggle, enable Settings → Accelerator → GPU T4 x2.")
VERSIONS = {"python": platform.python_version(), "llama_cpp_python": llama_cpp.__version__,
            "transformers": transformers.__version__, "datasets": datasets.__version__,
            "linguaprune": linguaprune.__version__}
print(VERSIONS, f"| GPU x{torch.cuda.device_count()}")

# %% [markdown]
# ## 1. The exam, rebuilt token for token as in notebook 07
#
# ---
#
# 🇷🇺 **1. Экзамен, собранный токен в токен как в ноутбуке 07**

# %%
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def encode(text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


LETTERS = ("A", "B", "C", "D", "E")
LETTER_IDS = [encode(" " + letter)[0] for letter in LETTERS]

exam, exam_meta = [], []  # exam: (ids, span or None, choice_ids or None)
for lang, df in load_eval_rows(EVAL_LANGUAGES, N_EVAL, SEED, reference_lang="en").items():
    for _, row in df.iterrows():
        ex = build_example(tokenizer, row)
        exam.append((ex["ids"][0].tolist(), ex["passage_span"], LETTER_IDS[:4]))
        exam_meta.append({"task": "belebele", "lang": lang, "label": ex["label"]})


def kazmmlu_rows(language: str, n: int) -> pd.DataFrame:
    tag = "kaz" if language == "kk" else "rus"
    frames = [load_dataset("MBZUAI/KazMMLU", subset, split="test").to_pandas()
              for subset in datasets.get_dataset_config_names("MBZUAI/KazMMLU") if subset.endswith(f"in {tag})")]
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
        exam.append((encode(text), None, [LETTER_IDS[LETTERS.index(l)] for l, _ in options]))
        exam_meta.append({"task": "kazmmlu", "lang": lang, "label": [o[0] for o in options].index(row["Answer Key"])})

for probe_id, (lang, prompt, correct, wrong) in enumerate(FACT_PROBES):
    prompt_ids = encode(prompt)
    for rank, answer in enumerate([correct, *wrong]):
        ids = prompt_ids + encode(answer)
        exam.append((ids, (len(prompt_ids), len(ids)), None))
        exam_meta.append({"task": "fact", "lang": lang, "probe": probe_id, "candidate": rank})

# the original model's scores from notebook 07 must line up with this exam row by row
scores_07 = pd.read_csv(SCORES_07)
original = scores_07[scores_07.config == "original"].reset_index(drop=True)
if len(original) != len(exam):
    raise RuntimeError(f"exam has {len(exam)} items, notebook 07 has {len(original)}")
for i, (ids, span, _) in enumerate(exam):
    m = exam_meta[i]
    if (original.task[i], original.lang[i]) != (m["task"], m["lang"]):
        raise RuntimeError(f"item {i}: task/lang differ from notebook 07")
    if span is not None and int(original.tokens[i]) != span[1] - span[0]:
        raise RuntimeError(f"item {i}: span length differs from notebook 07")
if max(len(ids) for ids, _, _ in exam) > N_CTX:
    raise RuntimeError("an exam item is longer than the context window")
print("Exam matches notebook 07:", pd.Series([m["task"] + "/" + m["lang"] for m in exam_meta]).value_counts().to_dict())

if DRY_RUN:  # a few items of every task, keeping the row order
    keep = [i for i, m in enumerate(exam_meta)
            if sum(1 for j in range(i) if exam_meta[j]["task"] == m["task"] and exam_meta[j]["lang"] == m["lang"]) < 6]
else:
    keep = list(range(len(exam)))

# %% [markdown]
# ## 2. Score every build with llama.cpp
#
# ---
#
# 🇷🇺 **2. Оцениваем каждую версию в llama.cpp**

# %%
def log_softmax(rows: np.ndarray) -> np.ndarray:
    rows = rows.astype(np.float64)
    top = rows.max(axis=-1, keepdims=True)
    return rows - top - np.log(np.exp(rows - top).sum(axis=-1, keepdims=True))


def score_item(llm: Llama, ids: list[int], span, choice_ids) -> tuple[float | None, list[float] | None]:
    llm.reset()
    llm.eval(ids)
    logits = np.asarray(llm.scores[: len(ids)])
    span_logprob = None
    if span is not None:
        start, end = span
        rows = log_softmax(logits[start - 1:end - 1])
        span_logprob = float(rows[np.arange(end - start), ids[start:end]].sum())
    choices = None if choice_ids is None else [float(logits[len(ids) - 1, c]) for c in choice_ids]
    return span_logprob, choices


def score_model(label: str, repo: str, filename: str) -> tuple[list[dict], dict]:
    started = time.time()
    path = Path(hf_hub_download(repo, filename, local_dir=DOWNLOAD_ROOT))
    size_gb = path.stat().st_size / 1e9
    try:
        llm = Llama(model_path=str(path), n_gpu_layers=-1, n_ctx=N_CTX, n_batch=512, logits_all=True, verbose=False)
        load_min = (time.time() - started) / 60
        started = time.time()
        model_rows = []
        for i in keep:
            ids, span, choice_ids = exam[i]
            span_logprob, choices = score_item(llm, ids, span, choice_ids)
            model_rows.append({"config": label, **exam_meta[i], "logprob": span_logprob,
                               "tokens": None if span is None else span[1] - span[0],
                               "choice": None if choices is None else int(np.argmax(choices))})
        del llm
    finally:
        path.unlink(missing_ok=True)
    info = {"repo": repo, "file": filename, "file_gb": size_gb, "load_min": load_min,
            "score_min": (time.time() - started) / 60}
    return model_rows, info


rows, model_info, failures = [], {}, {}
for label, repo, filename in COMPETITORS:
    try:
        model_rows, model_info[label] = score_model(label, repo, filename)
        rows.extend(model_rows)
        info = model_info[label]
        print(f"{label}: {info['file_gb']:.2f} GB, loaded in {info['load_min']:.1f} min, scored in {info['score_min']:.1f} min")
    except Exception:
        failures[label] = traceback.format_exc()
        print(f"{label}: FAILED\n{failures[label]}")
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "scores.csv", index=False)  # keep partial results if the session dies

# %% [markdown]
# ## 3. Results next to LinguaPrune (notebook 07)
#
# ---
#
# 🇷🇺 **3. Результаты рядом с LinguaPrune (ноутбук 07)**

# %%
ours = scores_07[scores_07.config.isin(["kk+ru 4/3-bit @90%", "kk+ru 4-bit"])].copy()
ours["config"] = "LinguaPrune " + ours.config
reference = original.iloc[keep].assign(config="original (16-bit)")
ours = pd.concat([ours[ours.config == c].reset_index(drop=True).iloc[keep] for c in ours.config.unique()])
df = pd.concat([reference, ours, pd.DataFrame(rows)], ignore_index=True)
summary_07 = json.loads(urllib.request.urlopen(SUMMARY_07).read())
size_gb = {"original (16-bit)": summary_07["fp16_gb"],
           "LinguaPrune kk+ru 4/3-bit @90%": summary_07["disk_gb"]["kk+ru 4/3-bit @90%"],
           "LinguaPrune kk+ru 4-bit": summary_07["disk_gb"]["kk+ru 4-bit"],
           **{label: info["file_gb"] for label, info in model_info.items()}}


def perplexity(group: pd.DataFrame) -> float:
    spans = group[group.task == "belebele"]
    return float(np.exp(-spans.logprob.sum() / spans.tokens.sum()))


table = []
for config, group in df.groupby("config", sort=False):
    row = {"config": config, "size_gb": size_gb[config]}
    for lang in ("kk", "ru", "en"):
        row[f"ppl_{lang}"] = perplexity(group[group.lang == lang])
        bele = group[(group.task == "belebele") & (group.lang == lang)]
        row[f"belebele_{lang}"] = float((bele.choice == bele.label).mean())
    for lang in ("kk", "ru"):
        mmlu = group[(group.task == "kazmmlu") & (group.lang == lang)]
        row[f"kazmmlu_{lang}"] = float((mmlu.choice == mmlu.label).mean())
    facts = group[group.task == "fact"]
    row["facts"] = float(np.mean([g.sort_values("candidate").logprob.values.argmax() == 0
                                  for _, g in facts.groupby("probe")]))
    table.append(row)
table = pd.DataFrame(table).set_index("config")
base = table.loc["original (16-bit)"]
for lang in ("kk", "ru", "en"):
    table[f"growth_{lang}"] = table[f"ppl_{lang}"] / base[f"ppl_{lang}"] - 1
table["compression"] = base.size_gb / table.size_gb
table = table.sort_values("size_gb", ascending=False)
pd.set_option("display.width", 220)
print(table[["size_gb", "compression", "growth_kk", "growth_ru", "growth_en", "kazmmlu_kk", "kazmmlu_ru",
             "belebele_kk", "belebele_ru", "belebele_en", "facts"]].round(3))

# %%
fig, ax = plt.subplots(figsize=(9, 5.5), dpi=140)
for config, r in table.drop(index="original (16-bit)").iterrows():
    mine = config.startswith("LinguaPrune")
    ax.scatter(r.size_gb, 100 * r.growth_kk, s=90 if mine else 45, color="#eb6834" if mine else "#2a78d6", zorder=3)
    ax.annotate(config.replace("LinguaPrune ", "LP "), (r.size_gb, 100 * r.growth_kk), textcoords="offset points",
                xytext=(6, 4), fontsize=8)
ax.axhline(2, color="#6b6a64", linestyle="--", linewidth=1)
ax.set_xlabel("Size on disk (GB)")
ax.set_ylabel("Kazakh perplexity growth vs. 16-bit (%)")
ax.set_title("Qwen3-30B-A3B: Kazakh quality vs. size (orange = LinguaPrune)", loc="left")
ax.grid(alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "kazakh_vs_size.png")
plt.show()

summary = {"versions": VERSIONS, "dry_run": DRY_RUN, "items_scored": len(keep), "models": model_info,
           "failures": failures,
           "table": json.loads(table.reset_index().to_json(orient="records"))}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("Done:", sorted(p.name for p in OUTPUT_DIR.iterdir()))
