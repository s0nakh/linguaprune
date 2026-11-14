# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 10 · A 2-bit GGUF tuned on Kazakh and Russian text
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# Notebook 09 showed that popular 2-bit GGUF files of Qwen3-30B-A3B lose 10–16% of Kazakh. llama.cpp chooses how to
# round each weight with an *importance matrix* (imatrix): statistics of the activations on some calibration text.
# Publishers compute it mostly on English. Here **only that text changes**; the model, the source file (`Q8_0`), the
# quantization type (`IQ2_M`) and the program stay the same:
#
# | Importance matrix | Text |
# |---|---|
# | kk+ru | 128 + 128 Kazakh and Russian Wikipedia articles (the notebook 07 calibration sampler) |
# | English (control) | 256 English Wikipedia articles, the same amount of text |
# | kk+ru+en | both of the above, combined |
# | publisher | `imatrix_unsloth.dat`, published next to the popular files |
#
# Every file takes the exam of notebooks 07 and 09 (same token ids, checked row by row).
#
# **Run:** Accelerator GPU T4 x2, Internet On. No secrets needed. Add the output of notebook 09 as input to reuse its
# llama-cpp-python build (otherwise it is rebuilt, ~30 min). The full run takes about 6 hours: each 2-bit file takes
# ~65 min to quantize on Kaggle's 4 CPU cores. `DRY_RUN = True`: one importance matrix on 4 chunks, one file, a few
# questions per task (~1.5 h).
#
# ---
#
# 🇷🇺 **10 · 2-битный GGUF, настроенный на казахском и русском тексте**
#
# Ноутбук 09 показал: популярные 2-битные файлы GGUF теряют 10–16% казахского. llama.cpp решает, как округлять каждый
# вес, по *матрице важности* (imatrix) — статистике активаций на калибровочном тексте. Издатели считают её в основном
# по английскому. Здесь **меняется только этот текст**; модель, исходный файл (`Q8_0`), тип сжатия (`IQ2_M`) и
# программа те же. Четыре матрицы: kk+ru, английская (контроль, столько же текста), kk+ru+en и матрица издателя.
# Каждый файл сдаёт экзамен ноутбуков 07 и 09.
#
# **Запуск:** GPU T4 x2, Internet On, секреты не нужны. Подключите вывод ноутбука 09 как входные данные, чтобы не
# пересобирать llama-cpp-python (~30 мин). Полный запуск — около 6 часов: каждый 2-битный файл сжимается ~65 мин на
# 4 ядрах процессора Kaggle. `DRY_RUN = True` — одна матрица на 4 кусках, один файл, несколько вопросов (~1,5 ч).

# %%
import glob
import os
import shutil
import subprocess
import sys
import tarfile
import threading
import time
from pathlib import Path

LINGUAPRUNE_REF = "8a88921b8cb8005c3c9b1f96d149635e68e6248f"  # pinned commit of the linguaprune package (v0.6.1)
LLAMA_CPP_PYTHON_VERSION = "0.3.36"  # the same llama.cpp as notebook 09, for scoring and for the tools
pip = [sys.executable, "-m", "pip", "install", "-q"]
subprocess.check_call(pip + [f"git+https://github.com/s0nakh/linguaprune@{LINGUAPRUNE_REF}"])

nvcc = shutil.which("nvcc") or next(iter(sorted(glob.glob("/usr/local/cuda*/bin/nvcc"))), None)
if nvcc is None:
    raise RuntimeError("No CUDA compiler (nvcc) found; enable Settings → Accelerator → GPU T4 x2.")
if shutil.which("cmake") is None:
    subprocess.check_call(pip + ["cmake", "ninja"])
KAGGLE_WORKING = Path("/kaggle/working")
WORK = Path("/kaggle/tmp") if KAGGLE_WORKING.exists() else Path("work")
WORK.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR = (KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")) / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

started = time.time()
wheels = sorted(glob.glob("/kaggle/input/**/llama_cpp_python-*.whl", recursive=True))
if not wheels:  # no notebook 09 output attached: build it as notebook 09 did
    build_env = {**os.environ, "CMAKE_ARGS": "-DGGML_CUDA=on -DCMAKE_CUDA_ARCHITECTURES=75 -DGGML_CUDA_NO_VMM=ON",
                 "FORCE_CMAKE": "1", "CUDACXX": nvcc, "PATH": f"{os.path.dirname(nvcc)}:{os.environ['PATH']}",
                 "CMAKE_BUILD_PARALLEL_LEVEL": str(os.cpu_count() or 4)}
    subprocess.check_call([sys.executable, "-m", "pip", "wheel", "-q", "--no-cache-dir", "--no-deps", "-w",
                           str(WORK / "wheels"), f"llama-cpp-python=={LLAMA_CPP_PYTHON_VERSION}"], env=build_env)
    wheels = sorted(glob.glob(str(WORK / "wheels" / "llama_cpp_python-*.whl")))
subprocess.check_call(pip + [wheels[0]])
print(f"llama-cpp-python ready in {(time.time() - started) / 60:.1f} min from {wheels[0]}")

# the llama.cpp source vendored in that same release, for llama-imatrix and llama-quantize
subprocess.check_call([sys.executable, "-m", "pip", "download", "-q", "--no-deps", "--no-binary", ":all:", "-d",
                       str(WORK / "sdist"), f"llama-cpp-python=={LLAMA_CPP_PYTHON_VERSION}"])
with tarfile.open(next((WORK / "sdist").glob("llama_cpp_python-*.tar.gz"))) as archive:
    archive.extractall(WORK / "sdist", filter="data")
LLAMA_SOURCE = next((WORK / "sdist").glob("llama_cpp_python-*/vendor/llama.cpp"))
print("llama.cpp source:", LLAMA_SOURCE)

# %%
import json
import platform
import traceback
import urllib.request

import datasets
import llama_cpp
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

import linguaprune
from linguaprune.exam import build_exam, check_against_reference, first_items_per_task, summarize
from linguaprune.llamacpp import (build_tools, imatrix_command, merge_imatrix_command, quantize_command, run_logged,
                                  score_gguf)

DRY_RUN = False  # True: one importance matrix on 4 chunks, one file, a few questions per task
BASE_MODEL = "Qwen/Qwen3-30B-A3B-Instruct-2507"
SOURCE_REPO = "unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF"
SOURCE_FILE = "Qwen3-30B-A3B-Instruct-2507-Q8_0.gguf"  # 32.5 GB; every file below is requantized from it
PUBLISHER_IMATRIX = "imatrix_unsloth.dat"
QUANT_TYPE = "IQ2_M"
REPO_RAW = "https://raw.githubusercontent.com/s0nakh/linguaprune/main/experiments"
DOCS_PER_LANGUAGE = 128  # Wikipedia articles per language, first 512 tokens each (the notebook 07 sampler)
CALIB_MAX_TOKENS = 512
SEED = 0
N_CTX = 4096
DRY_CHUNKS = 4
# label -> imatrix key, in the order they are built and scored (most important first)
ARMS = {"IQ2_M · kk+ru imatrix": "kkru", "IQ2_M · publisher imatrix": "publisher",
        "IQ2_M · kk+ru+en imatrix": "kkru_en", "IQ2_M · English imatrix": "en"}
OURS = ("kkru", "kkru_en")  # candidates for publishing; the best one on Kazakh is kept
if DRY_RUN:
    ARMS = {"IQ2_M · kk+ru imatrix": "kkru"}

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. On Kaggle, enable Settings → Accelerator → GPU T4 x2.")
VERSIONS = {"python": platform.python_version(), "llama_cpp_python": llama_cpp.__version__,
            "transformers": transformers.__version__, "datasets": datasets.__version__,
            "linguaprune": linguaprune.__version__}
free_gb = shutil.disk_usage(WORK).free / 1e9
print(VERSIONS, f"| GPU x{torch.cuda.device_count()} | CPU x{os.cpu_count()} | free disk {free_gb:.0f} GB")
if free_gb < 60:
    raise RuntimeError(f"Need ~60 GB of free disk (32.5 GB source + two 10 GB files + margin), have {free_gb:.0f} GB.")

# %% [markdown]
# ## 1. Tools and source file
#
# The 32.5 GB download runs while llama-imatrix and llama-quantize compile (~25 min).
#
# ---
#
# 🇷🇺 **1. Инструменты и исходный файл.** Файл на 32,5 ГБ скачивается, пока компилируются llama-imatrix и
# llama-quantize (~25 мин).

# %%
download = {}


def fetch_source():
    try:
        download["path"] = Path(hf_hub_download(SOURCE_REPO, SOURCE_FILE, local_dir=WORK))
        download["publisher_imatrix"] = Path(hf_hub_download(SOURCE_REPO, PUBLISHER_IMATRIX, local_dir=WORK))
    except Exception:
        download["error"] = traceback.format_exc()


started = time.time()
downloader = threading.Thread(target=fetch_source)
downloader.start()
LOGS = OUTPUT_DIR / "logs"
LOGS.mkdir(exist_ok=True)
tools = build_tools(LLAMA_SOURCE, WORK / "llama-build", nvcc, LOGS / "build.log")
print(f"tools built in {(time.time() - started) / 60:.1f} min")
if not DRY_RUN:  # static binaries, reused by later notebooks instead of a 20+ minute rebuild
    (KAGGLE_WORKING / "tools").mkdir(exist_ok=True)
    for tool in (tools.imatrix, tools.quantize):
        shutil.copy2(tool, KAGGLE_WORKING / "tools" / tool.name)
downloader.join()
if "error" in download:
    raise RuntimeError(download["error"])
SOURCE = download["path"]
print(f"source ready after {(time.time() - started) / 60:.1f} min: {SOURCE.stat().st_size / 1e9:.2f} GB")

# %% [markdown]
# ## 2. Calibration text and importance matrices
#
# ---
#
# 🇷🇺 **2. Калибровочный текст и матрицы важности**

# %%
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)


def wiki_text(config_name: str, docs: int, seed: int) -> tuple[str, int]:
    """The first 512 tokens of `docs` random articles (at least 500 characters), as in notebook 07."""
    stream = load_dataset("wikimedia/wikipedia", config_name, split="train", streaming=True)
    parts, tokens = [], 0
    for row in stream.shuffle(seed=seed, buffer_size=10_000):
        text = row["text"].strip()
        if len(text) < 500:
            continue
        ids = tokenizer(text, truncation=True, max_length=CALIB_MAX_TOKENS).input_ids
        parts.append(tokenizer.decode(ids))
        tokens += len(ids)
        if len(parts) == docs:
            return "\n\n".join(parts), tokens
    raise RuntimeError(f"not enough articles in {config_name}")


calibration = {}
kk, kk_tokens = wiki_text("20231101.kk", DOCS_PER_LANGUAGE, SEED + 1)
ru, ru_tokens = wiki_text("20231101.ru", DOCS_PER_LANGUAGE, SEED + 1)
en, en_tokens = wiki_text("20231101.en", 2 * DOCS_PER_LANGUAGE, SEED + 1)
for key, text in {"kkru": kk + "\n\n" + ru, "en": en}.items():
    calibration[key] = WORK / f"calibration_{key}.txt"
    calibration[key].write_text(text)
print(f"calibration tokens: kk {kk_tokens}, ru {ru_tokens}, en {en_tokens}")

needed = set(ARMS.values())
imatrix = {"publisher": download["publisher_imatrix"]}
imatrix_info = {}
for key in ("kkru", "en"):
    if key not in needed and not ("kkru_en" in needed):
        continue
    out = OUTPUT_DIR / f"imatrix_{key}.gguf"
    log = LOGS / f"imatrix_{key}.log"
    minutes = run_logged(imatrix_command(tools.imatrix, SOURCE, calibration[key], out,
                                         chunks=DRY_CHUNKS if DRY_RUN else None), log)
    text = log.read_text(errors="replace")
    imatrix[key] = out
    imatrix_info[key] = {"minutes": minutes, "partial_entries": text.count("has partial data"),
                         "skipped_entries": text.count("has no data")}
    print(f"imatrix {key}: {minutes:.1f} min, {imatrix_info[key]}")
if "kkru_en" in needed:
    imatrix["kkru_en"] = OUTPUT_DIR / "imatrix_kkru_en.gguf"
    run_logged(merge_imatrix_command(tools.imatrix, SOURCE, [imatrix["kkru"], imatrix["en"]], imatrix["kkru_en"]),
               LOGS / "imatrix_kkru_en.log")

# %% [markdown]
# ## 3. The exam, rebuilt token for token as in notebook 07
#
# ---
#
# 🇷🇺 **3. Экзамен, собранный токен в токен как в ноутбуке 07**

# %%
items = build_exam(tokenizer)
scores_07 = pd.read_csv(f"{REPO_RAW}/07/full/scores.csv")
original = scores_07[scores_07.config == "original"].reset_index(drop=True)
check_against_reference(items, original, N_CTX)
keep = first_items_per_task(items, 6) if DRY_RUN else list(range(len(items)))
print(f"exam matches notebook 07: {len(items)} items, scoring {len(keep)}")

# %% [markdown]
# ## 4. Quantize with each importance matrix and score
#
# ---
#
# 🇷🇺 **4. Сжимаем с каждой матрицей и оцениваем**

# %%
rows, file_info, failures, kept = [], {}, {}, {}
for label, key in ARMS.items():
    out = WORK / f"{key}-{QUANT_TYPE}.gguf"
    try:
        quantize_min = run_logged(quantize_command(tools.quantize, SOURCE, out, QUANT_TYPE, imatrix[key]),
                                  LOGS / f"quantize_{key}.log")
        model_rows, info = score_gguf(out, items, keep, label, N_CTX)
        rows.extend(model_rows)
        file_info[label] = {**info, "imatrix": key, "quantize_min": quantize_min}
        print(f"{label}: {info['file_gb']:.2f} GB, quantized in {quantize_min:.1f} min, "
              f"scored in {info['score_min']:.1f} min")
        if key in OURS and not DRY_RUN:
            kept[label] = out
    except Exception:
        failures[label] = traceback.format_exc()
        print(f"{label}: FAILED\n{failures[label]}")
    finally:
        if label not in kept:
            out.unlink(missing_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_DIR / "scores.csv", index=False)  # keep partial results if the session dies

# %% [markdown]
# ## 5. Results next to notebooks 07 and 09
#
# ---
#
# 🇷🇺 **5. Результаты рядом с ноутбуками 07 и 09**

# %%
scores_09 = pd.read_csv(f"{REPO_RAW}/09/scores.csv")
summary_07 = json.loads(urllib.request.urlopen(f"{REPO_RAW}/07/full/summary.json").read())
summary_09 = json.loads(urllib.request.urlopen(f"{REPO_RAW}/09/summary.json").read())
popular = ["UD-IQ2_M", "bartowski IQ2_M"]
reference = original.iloc[keep].assign(config="original (16-bit)")
theirs = pd.concat([scores_09[scores_09.config == c].reset_index(drop=True).iloc[keep] for c in popular])
df = pd.concat([reference, theirs, pd.DataFrame(rows)], ignore_index=True)
size_gb = {"original (16-bit)": summary_07["fp16_gb"],
           **{c: summary_09["models"][c]["file_gb"] for c in popular},
           **{label: info["file_gb"] for label, info in file_info.items()}}
table = summarize(df, size_gb, baseline="original (16-bit)")
pd.set_option("display.width", 220)
print(table[["size_gb", "compression", "growth_kk", "growth_ru", "growth_en", "kazmmlu_kk", "kazmmlu_ru",
             "belebele_kk", "belebele_ru", "belebele_en", "facts"]].round(3))

# %%
if kept:  # publish the candidate that loses the least Kazakh
    best = min(kept, key=lambda label: table.loc[label, "growth_kk"])
    (KAGGLE_WORKING / "gguf").mkdir(exist_ok=True)
    shutil.move(kept[best], KAGGLE_WORKING / "gguf" / f"Qwen3-30B-A3B-Instruct-2507-{ARMS[best]}-{QUANT_TYPE}.gguf")
    print(f"kept for publishing: {best}")

# %%
fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
shown = table.drop(index="original (16-bit)")
x = np.arange(len(shown))
for offset, lang, color in ((-0.27, "kk", "#E8622C"), (0, "ru", "#2A6FD6"), (0.27, "en", "#8A8A8A")):
    ax.bar(x + offset, 100 * shown[f"growth_{lang}"], width=0.26, label=lang, color=color)
ax.axhline(2, color="#666", linestyle="--", linewidth=1)
ax.set_xticks(x, [f"{c}\n{s:.1f} GB" for c, s in zip(shown.index, shown.size_gb)], fontsize=7.5)
ax.set_ylabel("Perplexity growth vs. 16-bit (%)")
ax.set_title("Qwen3-30B-A3B at 2 bits: which text the importance matrix saw", loc="left", fontsize=10)
ax.legend(frameon=False)
ax.grid(axis="y", alpha=0.3)
fig.tight_layout()
fig.savefig(OUTPUT_DIR / "imatrix_language.png")
plt.show()

summary = {"versions": VERSIONS, "dry_run": DRY_RUN, "items_scored": len(keep), "quant_type": QUANT_TYPE,
           "source": f"{SOURCE_REPO}/{SOURCE_FILE}",
           "calibration_tokens": {"kk": kk_tokens, "ru": ru_tokens, "en": en_tokens},
           "imatrix": imatrix_info, "files": file_info, "failures": failures,
           "table": json.loads(table.reset_index().to_json(orient="records"))}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print("Done:", sorted(p.name for p in OUTPUT_DIR.iterdir()))
