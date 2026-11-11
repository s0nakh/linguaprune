# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 04 · Keep every expert, compress the rarely used ones: language-aware mixed precision
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# Notebook 03 showed the limit of deleting experts: memory went down 38%, but a deleted expert takes
# its knowledge with it (Russian "Astana" became "Almaty"). Here **no expert is deleted**. Every expert
# gets a bit-width: the experts Kazakh and Russian use most stay at 4 bits, rarely used ones go down
# to 3 or 2 bits. Their knowledge is compressed, not removed.
#
# **Target, fixed before the run:** total memory **≥ 3.5× smaller** than fp16 (13.8 GB), while Belebele
# perplexity for **kk and ru grows by at most 2%**.
#
# **Compared at equal memory** (same number of experts per bit-width in every layer):
# our kk+ru ranking vs. English ranking vs. random, plus uniform 4/3/2-bit and "delete 16 experts + 4-bit".
#
# Quantization is simulated (weights are rounded to the low-bit grid and kept in fp16), which measures
# quality exactly; the memory column is what a real low-bit format needs, including scales.
#
# **Run:** GPU T4 x2, Internet On, **Save Version → Save & Run All (Commit)** (~45 min).
#
# ---
#
# 🇷🇺 **04 · Оставляем всех экспертов, редко используемых сжимаем: языковая смешанная точность**
#
# Ноутбук 03 показал предел удаления экспертов: память уменьшилась на 38%, но удалённый эксперт уносит
# свои знания («Астана» по-русски превратилась в «Алматы»). Здесь **ни один эксперт не удаляется**.
# Каждый получает свою точность: эксперты, которыми чаще пользуются казахский и русский, остаются
# в 4 битах, редко используемые сжимаются до 3 или 2 бит. Их знания сжимаются, а не стираются.
#
# **Цель, зафиксированная до прогона:** общая память **в ≥ 3,5 раза меньше** fp16 (13,8 ГБ), а перплексия
# Belebele для **kk и ru растёт не больше чем на 2%**.
#
# **Сравнение при одинаковой памяти** (одинаковое число экспертов каждой точности в каждом слое):
# наш порядок kk+ru против английского и случайного, плюс равномерные 4/3/2 бита и «удалить 16 экспертов + 4 бита».
#
# Квантование имитируется (веса округляются до сетки низкой точности и хранятся в fp16): так качество
# меряется точно, а столбец памяти показывает, сколько займёт настоящий низкобитный формат вместе с масштабами.
#
# **Запуск:** GPU T4 x2, Internet On, **Save Version → Save & Run All (Commit)** (~45 мин).

# %%
import subprocess
import sys

LINGUAPRUNE_REF = "v0.2.0"  # pinned release of the linguaprune package
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
from transformers import AutoModelForCausalLM, AutoTokenizer

import linguaprune
from linguaprune import (allocate_bits_by_coverage, allocate_like, count_parameters, expert_memory_bytes,
                         find_moe_blocks, language_frequency_score, masked_experts, quantize_experts_,
                         restore_experts, row_share, snapshot_experts, top_experts)
from linguaprune.belebele import answer_token_ids, build_example, evaluate_language, load_eval_rows

MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"  # same weights as notebooks 01–03
MODEL_DTYPE = torch.float16
GROUP_SIZE = 128
REPO_RAW = f"https://raw.githubusercontent.com/s0nakh/linguaprune/{LINGUAPRUNE_REF}"
CALIBRATION_URL = f"{REPO_RAW}/experiments/02/calibration.npz"
EVAL_LANGUAGES = {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}
N_EVAL = 150  # the same Belebele passages as notebooks 02–03
SEED = 0
RANDOM_SEEDS = (0, 1, 2)
TARGET_COMPRESSION = 3.5  # total memory must be at least this many times smaller than fp16
TARGET_MAX_PPL_GROWTH = 0.02  # kk and ru perplexity may grow by at most 2%
PROMPTS = {
    "ru_capital": "Столица Казахстана —",
    "kk_capital": "Қазақстанның астанасы —",
    "ru_abai": "Абай Кунанбаев — это",
    "en_capital": "The capital of Kazakhstan is",
}

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
# ## 1. Load the model, the calibration and the exam
#
# ---
#
# 🇷🇺 **1. Загружаем модель, калибровку и экзамен**

# %%
def download(url: str, path: Path) -> Path:
    try:
        urllib.request.urlretrieve(url, path)
    except OSError as error:
        raise RuntimeError(f"Could not download {url}. Is Internet enabled?") from error
    return path


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = AutoModelForCausalLM.from_pretrained(MODEL_ID, revision=MODEL_REVISION, dtype=MODEL_DTYPE,
                                             device_map="auto").eval()
offloaded = {k: v for k, v in getattr(model, "hf_device_map", {}).items() if str(v) in ("cpu", "disk")}
if model.dtype != MODEL_DTYPE or offloaded:
    raise RuntimeError(f"Model must be fully on GPU in fp16 (dtype {model.dtype}, offloaded {offloaded}).")
INPUT_DEVICE = model.get_input_embeddings().weight.device

calibration = np.load(download(CALIBRATION_URL, WORK_DIR / "nb02_calibration.npz"))
counts = {lang: calibration[f"{lang}_counts"] for lang in ("kk", "ru", "en")}
expected_shape = (model.config.num_hidden_layers, model.config.num_experts)
if any(c.shape != expected_shape for c in counts.values()):
    raise RuntimeError(f"Calibration shape does not match the model {expected_shape}.")

EVAL_ROWS = load_eval_rows(EVAL_LANGUAGES, N_EVAL, SEED, reference_lang="en")
EXAMPLES = {lang: [build_example(tokenizer, row) for _, row in df.iterrows()] for lang, df in EVAL_ROWS.items()}
ANSWER_IDS = answer_token_ids(tokenizer)

SNAPSHOT = snapshot_experts(model)  # original expert weights on CPU, restored before every configuration
expert_params = sum(b.experts.gate_up_proj.numel() + b.experts.down_proj.numel() for b in find_moe_blocks(model))
other_params = count_parameters(model) - expert_params
FP16_TOTAL_BYTES = count_parameters(model) * 2
OTHER_BYTES = other_params * 2  # attention, embeddings, norms stay fp16 in every configuration
print(f"Experts: {expert_params / 1e9:.2f}B params, other: {other_params / 1e9:.2f}B, "
      f"fp16 total {FP16_TOTAL_BYTES / 1e9:.1f} GB")

# %% [markdown]
# ## 2. Bit allocations
#
# - **kkru 4/3/2:** per layer, experts covering the first 60% of kk+ru router traffic → 4 bits,
#   up to 90% → 3 bits, the rarely used rest → 2 bits.
# - **kkru 4/3:** first 80% of traffic → 4 bits, the rest → 3 bits (no 2-bit experts).
# - **Controls** reuse exactly the same number of experts per bit-width, ranked by English traffic
#   or at random — so memory is identical and only the *choice* differs.
#
# ---
#
# 🇷🇺 **2. Распределение битов**
#
# - **kkru 4/3/2:** в каждом слое эксперты, на которых приходятся первые 60% трафика роутера kk+ru, → 4 бита,
#   до 90% → 3 бита, редко используемый остаток → 2 бита.
# - **kkru 4/3:** первые 80% трафика → 4 бита, остальное → 3 бита (без 2-битных экспертов).
# - **Контроли** берут ровно столько же экспертов каждой точности, но выбирают их по английскому трафику
#   или случайно — память одинаковая, отличается только *выбор*.

# %%
L, E = expected_shape
kkru_score = language_frequency_score(counts, ("kk", "ru"))
en_score = row_share(counts["en"])


def random_score(seed: int) -> np.ndarray:
    return np.random.default_rng([SEED, 200 + seed]).random((L, E))


bits_kkru_432 = allocate_bits_by_coverage(kkru_score, (0.6, 0.9), (4, 3, 2))
bits_kkru_43 = allocate_bits_by_coverage(kkru_score, (0.8,), (4, 3))
keep48 = top_experts(kkru_score, 48)

def deleted_as_zero(bits: np.ndarray, kept: np.ndarray) -> np.ndarray:
    """Memory view of a deletion config: deleted experts get 0 bits (no memory)."""
    is_kept = np.zeros((L, E), dtype=bool)
    np.put_along_axis(is_kept, kept, True, axis=1)
    return np.where(is_kept, bits, 0)


# name -> (bits used to quantize, bits used for memory accounting, kept experts if some are deleted)
CONFIGS = {
    "fp16 (original)": (None, None, None),
    "uniform 4-bit": (np.full((L, E), 4), np.full((L, E), 4), None),
    "uniform 3-bit": (np.full((L, E), 3), np.full((L, E), 3), None),
    "uniform 2-bit": (np.full((L, E), 2), np.full((L, E), 2), None),
    "kkru 4/3": (bits_kkru_43, bits_kkru_43, None),
    "en 4/3 (control)": (allocate_like(bits_kkru_43, en_score),) * 2 + (None,),
    "kkru 4/3/2": (bits_kkru_432, bits_kkru_432, None),
    "en 4/3/2 (control)": (allocate_like(bits_kkru_432, en_score),) * 2 + (None,),
    **{f"random 4/3/2 #{s}": (allocate_like(bits_kkru_432, random_score(s)),) * 2 + (None,) for s in RANDOM_SEEDS},
    "delete 16 + 4-bit": (np.full((L, E), 4), deleted_as_zero(np.full((L, E), 4), keep48), keep48),
}


def total_bytes(memory_bits: np.ndarray | None) -> float:
    if memory_bits is None:
        return float(FP16_TOTAL_BYTES)
    return OTHER_BYTES + expert_memory_bytes(model, memory_bits, GROUP_SIZE)


for name, (_, memory_bits, _) in CONFIGS.items():
    gb = total_bytes(memory_bits) / 1e9
    print(f"{name:>22}: {gb:5.2f} GB, {FP16_TOTAL_BYTES / 1e9 / gb:4.2f}× smaller than fp16")

# %% [markdown]
# ## 3. Main run
#
# For every configuration: restore the original experts, quantize, (optionally) mask deleted experts,
# then measure Belebele perplexity/accuracy and generate short answers to factual prompts.
# Results are appended to `results/results.csv` after each configuration.
#
# ---
#
# 🇷🇺 **3. Главный прогон**
#
# Для каждого варианта: возвращаем исходных экспертов, квантуем, (если нужно) маскируем удалённых,
# меряем перплексию и точность на Belebele и генерируем короткие ответы на фактические вопросы.
# После каждого варианта результаты дописываются в `results/results.csv`.

# %%
@torch.inference_mode()
def generate_answers(current_model) -> dict[str, str]:
    answers = {}
    for key, prompt in PROMPTS.items():
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(INPUT_DEVICE)
        out = current_model.generate(ids, max_new_tokens=20, do_sample=False)
        answers[key] = tokenizer.decode(out[0][ids.shape[1]:], skip_special_tokens=True).strip()
    return answers


def run_config(name: str, quant_bits, memory_bits, kept) -> tuple[list[dict], dict[str, str]]:
    restore_experts(model, SNAPSHOT)
    if quant_bits is not None:
        quantize_experts_(model, quant_bits, GROUP_SIZE)
    gb = total_bytes(memory_bits) / 1e9
    rows = []

    def measure():
        nonlocal rows
        for lang, examples in EXAMPLES.items():
            metrics = evaluate_language(model, examples, ANSWER_IDS, INPUT_DEVICE)
            rows = [*rows, {"config": name, "lang": lang, "memory_gb": gb, **metrics}]
        return generate_answers(model)

    if kept is None:
        answers = measure()
    else:
        with masked_experts(model, kept):
            answers = measure()
    return rows, answers


results, answers = [], {}
for number, (name, (quant_bits, memory_bits, kept)) in enumerate(CONFIGS.items(), start=1):
    started = time.time()
    rows, answers[name] = run_config(name, quant_bits, memory_bits, kept)
    results = [*results, *rows]
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "results.csv", index=False)
    took = time.time() - started
    print(f"[{number}/{len(CONFIGS)}] {name:>22} | {took:.0f} s | ~{took * (len(CONFIGS) - number) / 60:.0f} min left")
restore_experts(model, SNAPSHOT)

# %% [markdown]
# ## 4. Did we hit the target?
#
# ---
#
# 🇷🇺 **4. Достигнута ли цель?**

# %%
df = pd.DataFrame(results)
base = df[df.config == "fp16 (original)"].set_index("lang").perplexity
df["ppl_growth"] = df.apply(lambda r: r.perplexity / base[r.lang] - 1, axis=1)
table = df.pivot_table(index="config", columns="lang", values="ppl_growth").reindex(list(CONFIGS))
table["memory_gb"] = df.groupby("config").memory_gb.first().reindex(list(CONFIGS))
table["compression"] = FP16_TOTAL_BYTES / 1e9 / table["memory_gb"]
table["meets_target"] = ((table["compression"] >= TARGET_COMPRESSION)
                         & (table["kk"] <= TARGET_MAX_PPL_GROWTH) & (table["ru"] <= TARGET_MAX_PPL_GROWTH))
pd.set_option("display.width", 200)
print("Perplexity growth vs. fp16 (0.02 = +2%), memory and compression:")
print(table[["kk", "ru", "en", "memory_gb", "compression", "meets_target"]].round(3))
accuracy = df.pivot_table(index="config", columns="lang", values="accuracy").reindex(list(CONFIGS))
print("\nBelebele accuracy (chance = 0.25; weak signal for kk/ru on this model):")
print(accuracy.round(3))
winners = table.index[table.meets_target].tolist()
print(f"\nTarget (≥{TARGET_COMPRESSION}× smaller, kk/ru ≤ +{TARGET_MAX_PPL_GROWTH:.0%}): "
      f"{'MET by ' + ', '.join(winners) if winners else 'NOT met by any configuration'}")

# %%
print("Greedy answers to factual prompts:")
for key, prompt in PROMPTS.items():
    print(f"\n{prompt}")
    for name in CONFIGS:
        print(f"  {name:>22}: {answers[name][key][:90]!r}")

# %%
summary = {
    "model": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "versions": VERSIONS,
    "group_size": GROUP_SIZE,
    "quantization": "simulated asymmetric RTN, fp16 scale + zero per group; attention/embeddings kept fp16",
    "target": {"compression": TARGET_COMPRESSION, "max_ppl_growth_kk_ru": TARGET_MAX_PPL_GROWTH},
    "fp16_total_gb": FP16_TOTAL_BYTES / 1e9,
    "table": table.reset_index().to_dict(orient="records"),
    "accuracy": accuracy.reset_index().to_dict(orient="records"),
    "mean_bits": {name: float(mb.mean()) for name, (_, mb, _) in CONFIGS.items() if mb is not None},
    "bits_kkru_432": bits_kkru_432.tolist(),
    "bits_kkru_43": bits_kkru_43.tolist(),
    "answers": answers,
    "target_met_by": winners,
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
print("Done:", sorted(p.name for p in OUTPUT_DIR.iterdir()))
