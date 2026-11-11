# %% [markdown]
# # 01 · Какие эксперты MoE «говорят» на казахском языке
#
# *[English version](https://github.com/s0nakh/linguaprune/blob/main/notebooks/01_expert_profiling.ipynb) · часть проекта [LinguaPrune](https://github.com/s0nakh/linguaprune)*
#
# **Цель:** понять, какие эксперты модели-«смеси экспертов» (MoE) реально работают,
# когда модель читает текст на казахском языке, и сколько экспертов можно было бы убрать.
#
# **Модель:** `allenai/OLMoE-1B-7B-0125` (открытая, Apache 2.0, 64 эксперта в слое, 8 активных на токен).
# Это «тренажёр» перед AlemLLM: метод тот же, но модель помещается в бесплатный Kaggle.
#
# **Данные:** Википедия (`wikimedia/wikipedia`, снимок 2023-11-01) на казахском, русском, английском
# и китайском языках — только открытые данные, никаких клиентских. В китайской Википедии смешаны
# упрощённые и традиционные иероглифы.
#
# **Запуск на Kaggle:** Settings → Accelerator → **GPU T4 x2**, Internet → **On**, затем Run All.
# Результаты (картинки + json) появятся в `/kaggle/working/results`.
#
# **Чего этот ноутбук НЕ доказывает.** Мы считаем, *как часто* роутер выбирает эксперта.
# Частота ≠ важность: редкий эксперт может быть критичен для редких, но нужных токенов.
# Поэтому «покрыть 90% выборов» ≠ «сохранить 90% качества». Качество проверяется только в ноутбуке 02.

# %%
import json
import platform
import time
from functools import reduce
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import torch
import transformers
from datasets import load_dataset
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_DTYPE = torch.float16  # у T4 нет аппаратного bf16; fp16 + проверка на NaN/inf ниже
WIKI_DATASET = "wikimedia/wikipedia"
LANGUAGES = {  # код языка -> конфиг Википедии (фиксированный снимок)
    "kk": "20231101.kk",
    "ru": "20231101.ru",
    "en": "20231101.en",
    "zh": "20231101.zh",
}
TARGET_LANGUAGES = ("kk", "ru")  # языки, под которые «обрезаем» модель
TOKENS_PER_LANGUAGE = 100_000  # одинаковый бюджет токенов на каждый язык
MAX_TOKENS_PER_DOC = 512
MIN_CHARS_PER_DOC = 500
SHUFFLE_BUFFER = 10_000
COVERAGE_TARGET = 0.90  # сколько экспертов покрывают 90% выборов роутера
SPECIFICITY_MIN_SHARE = 1e-3  # эксперты с меньшей долей у обоих сравнений не раскрашиваем
SELF_TEST_MIN_AGREEMENT = 0.99  # доля токенов, где наш «жучок» совпал с выбором модели
BOOTSTRAP_ROUNDS = 200
SEED = 0

KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR = (KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")) / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("GPU не найдена. На Kaggle включи Settings → Accelerator → GPU T4 x2.")

torch.manual_seed(SEED)
RNG = np.random.default_rng(SEED)
VERSIONS = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
}
print(VERSIONS, f"| GPU x{torch.cuda.device_count()}")

# %% [markdown]
# ## 1. Загружаем модель и находим роутеры
#
# Роутер (`mlp.gate`) в каждом слое решает, каким экспертам отдать токен:
# он выдаёт 64 числа (логиты), и токен уходит к 8 экспертам с самыми большими.
# Мы ставим на роутер «жучок» (forward hook) и записываем эти числа.

# %%
def read_moe_config(config) -> tuple[int, int]:
    """Возвращает (число экспертов в слое, сколько экспертов активно на токен)."""
    num_experts = getattr(config, "num_experts", None) or getattr(config, "n_routed_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    if not num_experts or not top_k:
        raise ValueError(f"Не удалось прочитать MoE-параметры из конфига {type(config).__name__}")
    return int(num_experts), int(top_k)


def find_router_modules(model) -> list[torch.nn.Module]:
    routers = [module for name, module in model.named_modules() if name.endswith("mlp.gate")]
    if not routers:
        raise RuntimeError("Не найдено ни одного модуля 'mlp.gate' — это точно MoE-модель?")
    return routers


def load_model(model_id: str, revision: str):
    """Грузим ровно ту версию весов, номер которой запишем в отчёт, и проверяем, что всё на GPU."""
    loaded = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=MODEL_DTYPE, device_map="auto")
    if loaded.dtype != MODEL_DTYPE:
        raise RuntimeError(f"Модель загрузилась в {loaded.dtype}, а нужен {MODEL_DTYPE}.")
    offloaded = {k: v for k, v in getattr(loaded, "hf_device_map", {}).items() if str(v) in ("cpu", "disk")}
    if offloaded:
        raise RuntimeError(f"Часть модели не поместилась на GPU: {offloaded}")
    return loaded


MODEL_REVISION = HfApi().model_info(MODEL_ID).sha  # точная версия (коммит) весов на Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = load_model(MODEL_ID, MODEL_REVISION)
model.eval()

NUM_EXPERTS, TOP_K = read_moe_config(model.config)
ROUTERS = find_router_modules(model)
NUM_LAYERS = len(ROUTERS)
INPUT_DEVICE = model.get_input_embeddings().weight.device
print(f"Слоёв с роутером: {NUM_LAYERS}, экспертов в слое: {NUM_EXPERTS}, активных на токен: {TOP_K}")
print(f"Ревизия модели: {MODEL_REVISION}")

# %%
def extract_router_logits(output, num_experts: int) -> torch.Tensor:
    """Достаёт из выхода роутера тензор логитов [токены, эксперты]."""
    candidates = output if isinstance(output, (tuple, list)) else (output,)
    for tensor in candidates:
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.shape[-1] == num_experts:
            logits = tensor.reshape(-1, num_experts).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("NaN/inf в логитах роутера: fp16 переполнился.")
            return logits
    raise RuntimeError("Не нашли в выходе роутера тензор нужной формы — проверь версию transformers.")


def routing_stats(logits: torch.Tensor, num_experts: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """Для одного слоя: сколько раз выбран каждый эксперт и суммарная вероятность на него."""
    probs = torch.softmax(logits, dim=-1)
    chosen = probs.topk(top_k, dim=-1).indices.reshape(-1)
    counts = torch.bincount(chosen, minlength=num_experts).cpu().numpy()
    mass = probs.sum(dim=0).cpu().numpy()
    return counts, mass


class RouterRecorder:
    """Хранит последний выход каждого роутера; статистику считаем после прохода модели."""

    def __init__(self, routers: list[torch.nn.Module], num_experts: int):
        self.num_experts = num_experts
        self.latest: dict[int, torch.Tensor] = {}
        self.handles = [r.register_forward_hook(self._make_hook(i)) for i, r in enumerate(routers)]

    def _make_hook(self, layer_index: int):
        def hook(_module, _inputs, output):
            self.latest[layer_index] = extract_router_logits(output, self.num_experts).detach()
        return hook

    def pop(self) -> dict[int, torch.Tensor]:
        snapshot, self.latest = self.latest, {}
        return snapshot

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


# %% [markdown]
# ### Самопроверка «жучка»
#
# Прежде чем верить статистике, убедимся, что «жучок» видит те же решения, что и сама модель.
# Модель умеет сама отдавать логиты роутеров (`output_router_logits=True`) — сравниваем
# выбранных экспертов по каждому токену в каждом слое. Если не совпало — дальше не идём.

# %%
def topk_sets(logits: torch.Tensor, top_k: int) -> torch.Tensor:
    return logits.float().topk(top_k, dim=-1).indices.sort(dim=-1).values.cpu()


@torch.inference_mode()
def self_test_hooks(recorder: RouterRecorder, text: str) -> float:
    ids = tokenizer(text, return_tensors="pt").input_ids.to(INPUT_DEVICE)
    outputs = model(ids, use_cache=False, output_router_logits=True)
    hooked = recorder.pop()
    reference = outputs.router_logits
    if reference is None or len(reference) != NUM_LAYERS:
        raise RuntimeError("Модель не вернула логиты роутеров для всех слоёв.")
    agreements = [
        (topk_sets(hooked[i], TOP_K) == topk_sets(ref.reshape(-1, NUM_EXPERTS), TOP_K)).all(-1).float().mean()
        for i, ref in enumerate(reference)
    ]
    return float(torch.stack(agreements).min())


recorder = RouterRecorder(ROUTERS, NUM_EXPERTS)
agreement = self_test_hooks(recorder, "Қазақстан — Орталық Азиядағы мемлекет. Астанасы — Астана қаласы.")
print(f"Самопроверка: совпадение выбора экспертов (худший слой) = {agreement:.3f}")
if agreement < SELF_TEST_MIN_AGREEMENT:
    recorder.remove()
    raise RuntimeError("«Жучок» видит не то, что выбирает модель — статистике верить нельзя.")

# %% [markdown]
# ## 2. Прогоняем тексты на четырёх языках
#
# На каждый язык — одинаковый бюджет токенов, чтобы сравнение было честным.
# Первый токен каждой статьи не считаем: модель обрабатывает его нетипично
# («сток внимания», attention sink), и он исказил бы статистику.
#
# **Ограничение выборки.** `shuffle` в режиме стриминга перемешивает только буфер из
# 10 000 статей (и порядок файлов-шардов), поэтому мы видим не всю Википедию, а её кусок.
# Для оценки «на глаз» этого достаточно; для финальных цифр — параллельный корпус (FLORES, KazParC).

# %%
def iter_wiki_texts(wiki_config: str):
    try:
        stream = load_dataset(WIKI_DATASET, wiki_config, split="train", streaming=True)
    except Exception as error:  # сеть/имя конфига — частые причины
        raise RuntimeError(f"Не удалось открыть {WIKI_DATASET}/{wiki_config}. Internet включён?") from error
    for row in stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER):
        text = row["text"].strip()
        if len(text) >= MIN_CHARS_PER_DOC:
            yield text


@torch.inference_mode()
def profile_document(ids: torch.Tensor, recorder: RouterRecorder) -> tuple[np.ndarray, np.ndarray]:
    """Статистика роутеров по одной статье, без первого токена. Формы: [слои, эксперты]."""
    model(ids.to(INPUT_DEVICE), use_cache=False)
    layer_logits = recorder.pop()
    layer_stats = [routing_stats(layer_logits[i][1:], NUM_EXPERTS, TOP_K) for i in range(NUM_LAYERS)]
    return np.stack([c for c, _ in layer_stats]), np.stack([m for _, m in layer_stats])


def profile_language(wiki_config: str, recorder: RouterRecorder) -> dict:
    doc_counts, doc_mass = [], []
    tokens_seen = 0
    for text in iter_wiki_texts(wiki_config):
        budget_left = TOKENS_PER_LANGUAGE - tokens_seen + 1  # +1: первый токен не считается
        max_len = min(MAX_TOKENS_PER_DOC, budget_left)
        ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_len).input_ids
        if ids.shape[1] < 2:
            continue
        counts, mass = profile_document(ids, recorder)
        doc_counts = [*doc_counts, counts]
        doc_mass = [*doc_mass, mass]
        tokens_seen += ids.shape[1] - 1
        if tokens_seen >= TOKENS_PER_LANGUAGE:
            break
    return {
        "doc_counts": np.stack(doc_counts),  # [статьи, слои, эксперты]
        "mass": np.sum(doc_mass, axis=0),
        "tokens": tokens_seen,
        "docs": len(doc_counts),
    }


profiles: dict[str, dict] = {}
try:
    for lang, wiki_config in LANGUAGES.items():
        started = time.time()
        profiles = {**profiles, lang: profile_language(wiki_config, recorder)}
        p = profiles[lang]
        print(f"{lang}: {p['tokens']:,} токенов из {p['docs']} статей за {time.time() - started:.0f} с")
finally:
    recorder.remove()

# %% [markdown]
# ## 3. Анализ: сколько экспертов реально нужно казахскому языку
#
# Честность оценки обеспечиваем двумя приёмами из обычного ML:
# - **отложенная выборка:** экспертов выбираем по половине статей, а покрытие меряем на другой
#   половине (как train/test) — иначе оценка завышена;
# - **бутстрэп:** много раз пересэмплируем статьи и смотрим, как «гуляет» число нужных экспертов —
#   это даёт интервал вместо одной цифры.

# %%
def selection_share(counts: np.ndarray) -> np.ndarray:
    """Доля выборов каждого эксперта внутри слоя (строки суммируются в 1)."""
    return counts / counts.sum(axis=1, keepdims=True)


def experts_for_coverage(share: np.ndarray, target: float) -> list[np.ndarray]:
    """Для каждого слоя — минимальный набор экспертов, покрывающий `target` выборов."""
    kept = []
    for layer_share in share:
        order = np.argsort(-layer_share, kind="stable")
        cumulative = np.cumsum(layer_share[order])
        needed = min(int(np.searchsorted(cumulative, target)) + 1, len(order))
        kept.append(np.sort(order[:needed]))
    return kept


def union_for_targets(doc_counts: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Объединение наборов экспертов целевых языков, по слоям."""
    per_lang = [experts_for_coverage(selection_share(doc_counts[lang].sum(0)), COVERAGE_TARGET)
                for lang in TARGET_LANGUAGES]
    return [reduce(np.union1d, [kept[layer] for kept in per_lang]) for layer in range(NUM_LAYERS)]


def covered_share(share: np.ndarray, kept: list[np.ndarray]) -> float:
    """Средняя по слоям доля выборов, которая попадает в оставленных экспертов."""
    return float(np.mean([share[layer, k].sum() for layer, k in enumerate(kept)]))


def specificity(share: dict[str, np.ndarray], lang: str) -> np.ndarray:
    """log2(доля у языка / средняя доля у остальных): >0 — эксперт чаще выбирается на этом языке."""
    others = np.mean([s for other, s in share.items() if other != lang], axis=0)
    rare = (share[lang] < SPECIFICITY_MIN_SHARE) & (others < SPECIFICITY_MIN_SHARE)
    ratio = np.log2(np.maximum(share[lang], SPECIFICITY_MIN_SHARE) / np.maximum(others, SPECIFICITY_MIN_SHARE))
    return np.where(rare, np.nan, ratio)


doc_counts = {lang: p["doc_counts"] for lang, p in profiles.items()}
shares = {lang: selection_share(c.sum(0)) for lang, c in doc_counts.items()}
mass_shares = {lang: selection_share(p["mass"]) for lang, p in profiles.items()}
needed_per_layer = {lang: np.array([len(k) for k in experts_for_coverage(s, COVERAGE_TARGET)])
                    for lang, s in shares.items()}

target_union = union_for_targets(doc_counts)
kept_for_targets = np.array([len(k) for k in target_union])
prunable_fraction = 1 - kept_for_targets.mean() / NUM_EXPERTS

print(f"Экспертов, нужных для {COVERAGE_TARGET:.0%} выборов роутера (в среднем по слоям, из {NUM_EXPERTS}):")
for lang, needed in needed_per_layer.items():
    print(f"  {lang}: {needed.mean():.1f}")
print(f"\nДля {' + '.join(TARGET_LANGUAGES)} вместе нужно {kept_for_targets.mean():.1f} из {NUM_EXPERTS}")
print(f"→ оценка: можно убрать ~{prunable_fraction:.0%} экспертов (гипотеза; качество проверим в ноутбуке 02)")

# %%
def holdout_coverage() -> dict[str, float]:
    """Выбираем экспертов по чётным статьям, меряем покрытие на нечётных."""
    train = {lang: doc_counts[lang][0::2] for lang in TARGET_LANGUAGES}
    kept = union_for_targets(train)
    return {lang: covered_share(selection_share(doc_counts[lang][1::2].sum(0)), kept)
            for lang in TARGET_LANGUAGES}


def bootstrap_kept(rounds: int) -> np.ndarray:
    """Среднее по слоям число оставленных экспертов на бутстрэп-выборках статей."""
    results = []
    for _ in range(rounds):
        resampled = {lang: c[RNG.integers(0, len(c), len(c))] for lang, c in doc_counts.items()
                     if lang in TARGET_LANGUAGES}
        results = [*results, np.mean([len(k) for k in union_for_targets(resampled)])]
    return np.array(results)


holdout = holdout_coverage()
boot = bootstrap_kept(BOOTSTRAP_ROUNDS)
boot_low, boot_high = np.percentile(boot, [2.5, 97.5])
mass_coverage = {lang: covered_share(mass_shares[lang], target_union) for lang in TARGET_LANGUAGES}

print("Покрытие на ОТЛОЖЕННЫХ статьях (цель — " f"{COVERAGE_TARGET:.0%}):",
      {k: f"{v:.1%}" for k, v in holdout.items()})
print(f"Оставить экспертов, 95% бутстрэп-интервал: {boot_low:.1f}–{boot_high:.1f} из {NUM_EXPERTS}")
print("Покрытие по вероятностной массе роутера (а не по числу выборов):",
      {k: f"{v:.1%}" for k, v in mass_coverage.items()})

# %% [markdown]
# ## 4. Картинки
#
# Важно: различия между языками здесь — смесь «языка» и «темы». Статьи Википедии на разных
# языках пишут о разном (в казахской больше статей про сёла Казахстана), поэтому «казахский»
# эксперт может оказаться экспертом по географии. Развести это поможет параллельный корпус.

# %%
share_max = max(s.max() for s in shares.values())  # общая шкала — панели можно сравнивать
fig, axes = plt.subplots(1, len(shares), figsize=(5 * len(shares), 4), sharey=True)
for ax, (lang, share) in zip(axes, shares.items()):
    im = ax.imshow(share, aspect="auto", cmap="viridis", vmin=0, vmax=share_max)
    ax.set_title(f"{lang}: доля выборов эксперта")
    ax.set_xlabel("эксперт")
axes[0].set_ylabel("слой")
fig.colorbar(im, ax=axes, shrink=0.8)
fig.savefig(OUTPUT_DIR / "expert_share_by_language.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
kk_specificity = specificity(shares, "kk")
limit = np.nanmax(np.abs(kk_specificity))
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(kk_specificity, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
ax.set_title("kk против среднего по ru/en/zh, log2: красный — чаще на казахском, синий — реже\n"
             "(язык и тема статей смешаны; серое/пустое — редкие эксперты)")
ax.set_xlabel("эксперт")
ax.set_ylabel("слой")
fig.colorbar(im, ax=ax)
fig.savefig(OUTPUT_DIR / "kk_specificity.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 4))
for lang, needed in needed_per_layer.items():
    ax.plot(needed, marker="o", label=lang)
ax.plot(kept_for_targets, "k--", label=" + ".join(TARGET_LANGUAGES) + " (объединение)")
ax.axhline(NUM_EXPERTS, color="gray", lw=0.8)
ax.set_title(f"Сколько экспертов нужно для {COVERAGE_TARGET:.0%} выборов роутера")
ax.set_xlabel("слой")
ax.set_ylabel("экспертов")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))  # легенда справа, не закрывает линии
fig.savefig(OUTPUT_DIR / "experts_needed_per_layer.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Сохраняем результаты (понадобятся для ноутбука 02 — обрезка и проверка качества)

# %%
np.savez_compressed(
    OUTPUT_DIR / "routing_counts.npz",
    **{f"{lang}_doc_counts": p["doc_counts"] for lang, p in profiles.items()},
    **{f"{lang}_mass": p["mass"] for lang, p in profiles.items()},
)
summary = {
    "model": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dtype": str(MODEL_DTYPE),
    "versions": VERSIONS,
    "seed": SEED,
    "dataset": WIKI_DATASET,
    "dataset_configs": LANGUAGES,
    "num_layers": NUM_LAYERS,
    "num_experts": NUM_EXPERTS,
    "top_k": TOP_K,
    "coverage_target": COVERAGE_TARGET,
    "self_test_agreement": agreement,
    "tokens": {lang: p["tokens"] for lang, p in profiles.items()},
    "docs": {lang: p["docs"] for lang, p in profiles.items()},
    "mean_experts_needed": {lang: float(n.mean()) for lang, n in needed_per_layer.items()},
    "target_languages": list(TARGET_LANGUAGES),
    "kept_experts_per_layer": [k.tolist() for k in target_union],
    "estimated_prunable_fraction": float(prunable_fraction),
    "holdout_coverage": holdout,
    "mass_coverage": mass_coverage,
    "kept_mean_bootstrap_95ci": [float(boot_low), float(boot_high)],
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
print(json.dumps({k: v for k, v in summary.items() if k != "kept_experts_per_layer"}, ensure_ascii=False, indent=2))
