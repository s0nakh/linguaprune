# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 03 · Physically remove experts and save a smaller model
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# Notebook 02 only *simulated* removal: experts stayed in the model and the router was blocked
# from picking them. Here we **actually delete** their weights, save a smaller model that loads with
# the standard `from_pretrained`, and prove that it behaves exactly like the simulation.
#
# **Plan:**
# 1. Load OLMoE (the same weight revision as notebook 02).
# 2. Choose experts with notebook 02's kk+ru calibration: the 38 most-used experts per layer.
# 3. Measure the *simulated* model (mask) as a reference.
# 4. Delete the other experts' weights and check that the outputs match the reference.
# 5. Save the smaller model, load it back from disk and check it again.
#
# **Run:** GPU T4 x2, Internet On, then **Save Version → Save & Run All (Commit)** (~30–40 min).
# The smaller model (~8.6 GB) appears in Output → `linguaprune-olmoe-kkru-k38`.
#
# ---
#
# 🇷🇺 **03 · Физически удаляем экспертов и сохраняем модель меньшего размера**
#
# Ноутбук 02 только *имитировал* увольнение: эксперты оставались в модели, роутеру просто запрещали
# их выбирать. Здесь мы **по-настоящему удаляем** их веса, сохраняем уменьшенную модель, которая
# загружается обычным `from_pretrained`, и доказываем, что она ведёт себя точно как имитация.
#
# **План:**
# 1. Загружаем OLMoE (ту же версию весов, что в ноутбуке 02).
# 2. Выбираем экспертов по калибровке kk+ru из ноутбука 02: 38 самых используемых в каждом слое.
# 3. Меряем *имитацию* (маску) как эталон.
# 4. Удаляем веса остальных экспертов и проверяем, что ответы совпадают с эталоном.
# 5. Сохраняем модель, загружаем её с диска заново и проверяем ещё раз.
#
# **Запуск:** GPU T4 x2, Internet On, затем **Save Version → Save & Run All (Commit)** (~30–40 мин).
# Уменьшенная модель (~8,6 ГБ) появится в Output → `linguaprune-olmoe-kkru-k38`.

# %%
import subprocess
import sys

LINGUAPRUNE_REF = "v0.1.0"  # pinned release of the linguaprune package
subprocess.check_call([sys.executable, "-m", "pip", "install", "-q",
                       f"git+https://github.com/s0nakh/linguaprune@{LINGUAPRUNE_REF}"])

# %%
import gc
import json
import platform
import shutil
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
from linguaprune import (count_parameters, language_frequency_score, masked_experts, prune_experts,
                         top_experts)
from linguaprune.belebele import answer_token_ids, build_example, evaluate_language, load_eval_rows

MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_REVISION = "9b0c1aa87e34a20052389dce1f0cf01da783f654"  # same weights as notebooks 01 and 02
MODEL_DTYPE = torch.float16  # T4 has no hardware bf16
TARGET_LANGUAGES = ("kk", "ru")
KEEP = 38  # experts kept per layer (of 64); change to 48 for a milder cut
REPO_RAW = f"https://raw.githubusercontent.com/s0nakh/linguaprune/{LINGUAPRUNE_REF}"
CALIBRATION_URL = f"{REPO_RAW}/experiments/02/calibration.npz"
NB02_RESULTS_URL = f"{REPO_RAW}/experiments/02/results.csv"
EVAL_LANGUAGES = {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}
N_EVAL = 150  # the same Belebele passages as notebook 02
SEED = 0
CONFIDENT_MARGIN = 2.0  # a position is "confident" if the reference top-1 logit leads the runner-up by ≥2
MIN_CONFIDENT_AGREEMENT = 0.999  # where the model is confident, both models must predict the same token
MAX_PPL_REL_DIFF = 0.005  # perplexity of the pruned model must match the simulation within 0.5%
MAX_MEAN_LOGIT_DIFF = 0.05  # mean |difference| of logits; the max is only reported, since fp16 noise
                            # can flip one near-tied expert on one token and move that position a lot
PROMPTS = {
    "kk": "Қазақстанның астанасы",
    "ru": "Столица Казахстана —",
    "en": "The capital of Kazakhstan is",
}

KAGGLE_WORKING = Path("/kaggle/working")
WORK_DIR = KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")
OUTPUT_DIR = WORK_DIR / "results"
MODEL_DIR = WORK_DIR / f"linguaprune-olmoe-kkru-k{KEEP}"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. On Kaggle, enable Settings → Accelerator → GPU T4 x2.")

torch.manual_seed(SEED)
VERSIONS = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
    "linguaprune": linguaprune.__version__,
}
print(VERSIONS, f"| GPU x{torch.cuda.device_count()}")
print(f"linguaprune {linguaprune.__version__} from {linguaprune.__file__} (pinned ref {LINGUAPRUNE_REF})")

# %% [markdown]
# ## 1. Load the original model
#
# ---
#
# 🇷🇺 **1. Загружаем исходную модель**

# %%
def load_model(path_or_id: str, revision: str | None = None):
    """Load in fp16 on GPU and check that nothing was offloaded to CPU or disk."""
    loaded = AutoModelForCausalLM.from_pretrained(path_or_id, revision=revision, dtype=MODEL_DTYPE, device_map="auto")
    if loaded.dtype != MODEL_DTYPE:
        raise RuntimeError(f"Model loaded as {loaded.dtype}, expected {MODEL_DTYPE}.")
    offloaded = {k: v for k, v in getattr(loaded, "hf_device_map", {}).items() if str(v) in ("cpu", "disk")}
    if offloaded:
        raise RuntimeError(f"Part of the model did not fit on GPU: {offloaded}")
    return loaded.eval()


tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = load_model(MODEL_ID, MODEL_REVISION)
INPUT_DEVICE = model.get_input_embeddings().weight.device
print(f"Experts per layer: {model.config.num_experts}, active: {model.config.num_experts_per_tok}, "
      f"parameters: {count_parameters(model) / 1e9:.2f}B")

# %% [markdown]
# ## 2. Choose the experts to keep
#
# We reuse the router statistics that notebook 02 collected on 100k Wikipedia tokens per language,
# and keep the `KEEP` experts that Kazakh and Russian use most in every layer (the `freq_kkru` method).
#
# ---
#
# 🇷🇺 **2. Выбираем, каких экспертов оставить**
#
# Берём статистику роутера, которую ноутбук 02 собрал на 100 тыс. токенов Википедии на каждом языке,
# и в каждом слое оставляем `KEEP` экспертов, которых чаще всего используют казахский и русский
# (метод `freq_kkru`).

# %%
def download(url: str, path: Path) -> Path:
    try:
        urllib.request.urlretrieve(url, path)
    except OSError as error:
        raise RuntimeError(f"Could not download {url}. Is Internet enabled?") from error
    return path


calibration = np.load(download(CALIBRATION_URL, WORK_DIR / "nb02_calibration.npz"))
counts = {lang: calibration[f"{lang}_counts"] for lang in TARGET_LANGUAGES}
expected_shape = (model.config.num_hidden_layers, model.config.num_experts)
if any(c.shape != expected_shape for c in counts.values()):
    raise RuntimeError(f"Calibration shape {[c.shape for c in counts.values()]} does not match the model {expected_shape}.")
KEPT = top_experts(language_frequency_score(counts, TARGET_LANGUAGES), KEEP)
print(f"Keeping {KEEP} of {model.config.num_experts} experts in each of {KEPT.shape[0]} layers")

# %% [markdown]
# ## 3. Reference: the original model and the simulated removal (mask)
#
# On the same 150 Belebele passages as notebook 02 we measure the original model and the mask.
# We also keep the mask's raw outputs on a few texts to compare with the pruned model token by token.
#
# ---
#
# 🇷🇺 **3. Эталон: исходная модель и имитация увольнения (маска)**
#
# На тех же 150 отрывках Belebele, что в ноутбуке 02, меряем исходную модель и маску.
# Ещё сохраняем «сырые» выходы маски на нескольких текстах, чтобы потом сравнить их с уменьшенной
# моделью токен за токеном.

# %%
EVAL_ROWS = load_eval_rows(EVAL_LANGUAGES, N_EVAL, SEED, reference_lang="en")
EXAMPLES = {lang: [build_example(tokenizer, row) for _, row in df.iterrows()] for lang, df in EVAL_ROWS.items()}
ANSWER_IDS = answer_token_ids(tokenizer)
PROBE_IDS = [ex["ids"] for lang in EVAL_LANGUAGES for ex in EXAMPLES[lang][:3]]  # 9 texts for logit checks


def evaluate_all(current_model) -> dict[str, dict]:
    return {lang: evaluate_language(current_model, examples, ANSWER_IDS, INPUT_DEVICE)
            for lang, examples in EXAMPLES.items()}


@torch.inference_mode()
def probe_logits(current_model) -> list[torch.Tensor]:
    return [current_model(ids.to(INPUT_DEVICE), use_cache=False).logits[0].float().cpu() for ids in PROBE_IDS]


@torch.inference_mode()
def generate_samples(current_model) -> dict[str, str]:
    samples = {}
    for lang, prompt in PROMPTS.items():
        ids = tokenizer(prompt, return_tensors="pt").input_ids.to(INPUT_DEVICE)
        out = current_model.generate(ids, max_new_tokens=30, do_sample=False)
        samples[lang] = tokenizer.decode(out[0], skip_special_tokens=True)
    return samples


started = time.time()
base_metrics = evaluate_all(model)
base_samples = generate_samples(model)
with masked_experts(model, KEPT):
    mask_metrics = evaluate_all(model)
    mask_logits = probe_logits(model)
print(f"Reference measured in {time.time() - started:.0f} s")
print(pd.DataFrame({"original": {l: m["perplexity"] for l, m in base_metrics.items()},
                    "mask": {l: m["perplexity"] for l, m in mask_metrics.items()}}).round(3))

# %% [markdown]
# ## 4. Delete the removed experts' weights
#
# `prune_experts` cuts the rows of removed experts out of every router and every expert weight
# tensor, and sets `num_experts = KEEP` in the config. Then the pruned model must match the mask:
# the same perplexity (within 0.5%) and the same next-token prediction wherever the model is confident.
# Where two tokens are almost tied, fp16 rounding may swap them — that is noise, not a bug.
#
# ---
#
# 🇷🇺 **4. Удаляем веса уволенных экспертов**
#
# `prune_experts` вырезает строки уволенных экспертов из каждого роутера и каждого тензора весов
# экспертов и ставит в конфиге `num_experts = KEEP`. После этого уменьшенная модель должна совпасть
# с маской: та же перплексия (в пределах 0,5%) и тот же следующий токен везде, где модель уверена.
# Где два токена почти равны, округление fp16 может поменять их местами: это шум, а не ошибка.

# %%
def compare_logits(reference: list[torch.Tensor], candidate: list[torch.Tensor]) -> dict[str, float]:
    """Logit differences and top-1 agreement, overall and on positions where the reference is confident.

    fp16 rounding differs slightly between the masked and the pruned model (different matrix shapes).
    Where two next tokens are almost tied, that noise can swap them — harmless. Where the model is
    confident (top-1 leads by ≥ CONFIDENT_MARGIN logits), a swap would signal a real bug.
    """
    ref, cand = torch.cat(reference), torch.cat(candidate)
    same = ref.argmax(-1) == cand.argmax(-1)
    top2 = ref.topk(2, dim=-1).values
    confident = (top2[:, 0] - top2[:, 1]) >= CONFIDENT_MARGIN
    diff = (ref - cand).abs()
    return {"max_logit_diff": diff.max().item(),
            "mean_logit_diff": diff.mean().item(),
            "top1_agreement": same.float().mean().item(),
            "confident_share": confident.float().mean().item(),
            "confident_top1_agreement": same[confident].float().mean().item() if confident.any() else float("nan")}


report = prune_experts(model, KEPT)
torch.cuda.empty_cache()
print(f"Experts per layer: {report.experts_before} → {report.experts_after}")
print(f"Parameters: {report.params_before / 1e9:.2f}B → {report.params_after / 1e9:.2f}B "
      f"(−{report.removed_fraction:.0%}), fp16 size ≈ {report.params_after * 2 / 1e9:.1f} GB")

pruned_logits = probe_logits(model)
vs_mask = compare_logits(mask_logits, pruned_logits)
pruned_metrics = evaluate_all(model)
ppl_rel_diff = max(abs(pruned_metrics[l]["perplexity"] / mask_metrics[l]["perplexity"] - 1) for l in EVAL_LANGUAGES)
print(f"Pruned vs. mask: max perplexity difference {ppl_rel_diff:.4%} (≤{MAX_PPL_REL_DIFF:.1%}), "
      f"mean logit diff {vs_mask['mean_logit_diff']:.4f} (≤{MAX_MEAN_LOGIT_DIFF}), "
      f"top-1 agreement on confident positions {vs_mask['confident_top1_agreement']:.4f} "
      f"(≥{MIN_CONFIDENT_AGREEMENT}; {vs_mask['confident_share']:.0%} of positions are confident)")
print(f"  reported only: top-1 agreement on all positions {vs_mask['top1_agreement']:.4f}, "
      f"max logit diff {vs_mask['max_logit_diff']:.3f}")
checks_passed = (ppl_rel_diff <= MAX_PPL_REL_DIFF and vs_mask["mean_logit_diff"] <= MAX_MEAN_LOGIT_DIFF
                 and vs_mask["confident_top1_agreement"] >= MIN_CONFIDENT_AGREEMENT)  # NaN fails
if not checks_passed:
    raise RuntimeError("The pruned model does not match the simulated removal — do not trust or save it.")

# %% [markdown]
# ## 5. Save the smaller model and load it back from disk
#
# A fresh `from_pretrained` of the saved folder must give the same outputs as the model in memory.
# This proves the checkpoint is complete and usable on its own.
#
# ---
#
# 🇷🇺 **5. Сохраняем уменьшенную модель и загружаем её с диска заново**
#
# Новая загрузка сохранённой папки через `from_pretrained` должна давать те же выходы, что и модель
# в памяти. Это доказывает, что сохранённая модель полная и работает сама по себе.

# %%
MODEL_CARD = f"""---
license: apache-2.0
base_model: {MODEL_ID}
language: [kk, ru, en]
tags: [mixture-of-experts, pruning, linguaprune]
---
# LinguaPrune · OLMoE-1B-7B pruned for Kazakh + Russian ({KEEP}/64 experts per layer)

Experts were chosen by router usage on Kazakh and Russian Wikipedia; the other experts were deleted.
Made with [LinguaPrune](https://github.com/s0nakh/linguaprune) by Sophia Akhmetova.

Research artifact: perplexity on Belebele grows vs. the original model (see the repository for numbers);
English quality is expected to drop; the base model is weak in Kazakh. Not quantized yet.\nBase weights: `{MODEL_ID}` @ `{MODEL_REVISION}`.

🇷🇺 OLMoE-1B-7B, уменьшенная под казахский и русский ({KEEP} из 64 экспертов в слое).
Исследовательский артефакт: цифры качества — в репозитории LinguaPrune; модель ещё не квантована.
"""

if MODEL_DIR.exists():
    shutil.rmtree(MODEL_DIR)
model.save_pretrained(MODEL_DIR, max_shard_size="2GB")
tokenizer.save_pretrained(MODEL_DIR)
(MODEL_DIR / "README.md").write_text(MODEL_CARD, encoding="utf-8")
size_gb = sum(f.stat().st_size for f in MODEL_DIR.rglob("*") if f.is_file()) / 1e9
print(f"Saved to {MODEL_DIR} ({size_gb:.1f} GB)")

del model  # free the in-memory copy; its outputs are kept in `pruned_logits` and `pruned_metrics`
gc.collect()
torch.cuda.empty_cache()
reloaded = load_model(str(MODEL_DIR))
if reloaded.config.num_experts != KEEP:
    raise RuntimeError(f"Reloaded config has {reloaded.config.num_experts} experts, expected {KEEP}.")
vs_memory = compare_logits(pruned_logits, probe_logits(reloaded))
print(f"Reloaded vs. in-memory: top-1 agreement {vs_memory['top1_agreement']:.4f}, "
      f"mean logit diff {vs_memory['mean_logit_diff']:.2e}, max {vs_memory['max_logit_diff']:.2e}")
if vs_memory["top1_agreement"] < 0.999 or vs_memory["mean_logit_diff"] > 1e-3:
    raise RuntimeError("The model loaded from disk differs from the one in memory — the checkpoint is broken.")
pruned_samples = generate_samples(reloaded)

# %% [markdown]
# ## 6. Results
#
# ---
#
# 🇷🇺 **6. Результаты**

# %%
nb02 = pd.read_csv(download(NB02_RESULTS_URL, WORK_DIR / "nb02_results.csv"))
nb02_base = nb02[nb02.method == "base"].set_index("lang").perplexity
nb02_ours = nb02[(nb02.method == "freq_kkru") & (nb02.keep == KEEP)].set_index("lang").perplexity
table = pd.DataFrame({
    "original": {l: m["perplexity"] for l, m in base_metrics.items()},
    "mask": {l: m["perplexity"] for l, m in mask_metrics.items()},
    "pruned": {l: m["perplexity"] for l, m in pruned_metrics.items()},
})
table["pruned / original"] = table["pruned"] / table["original"]
table["nb02 ratio"] = nb02_ours / nb02_base if not nb02_ours.empty else np.nan
print("Belebele perplexity (lower = better):")
print(table.round(3))
for lang in PROMPTS:
    print(f"\n[{lang}] original: {base_samples[lang]}\n[{lang}] pruned:   {pruned_samples[lang]}")

summary = {
    "model": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "versions": VERSIONS,
    "keep": KEEP,
    "target_languages": list(TARGET_LANGUAGES),
    "params_before": report.params_before,
    "params_after": report.params_after,
    "removed_param_fraction": report.removed_fraction,
    "saved_size_gb": size_gb,
    "pruned_vs_mask": {**vs_mask, "max_ppl_rel_diff": ppl_rel_diff},
    "reloaded_vs_memory": vs_memory,
    "perplexity": table.drop(columns=["nb02 ratio"]).to_dict(orient="index"),
    "nb02_ratio": table["nb02 ratio"].to_dict(),
    "kept_experts_per_layer": KEPT.tolist(),
    "samples": {"original": base_samples, "pruned": pruned_samples},
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2, default=float))
print("\nDone:", sorted(p.name for p in OUTPUT_DIR.iterdir()))
