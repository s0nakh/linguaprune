# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 02 · «Увольняем» экспертов и проверяем, не поглупела ли модель
#
# **Автор:** София Ахметова (Sophia Akhmetova, [@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# *[English version](https://github.com/s0nakh/linguaprune/blob/main/notebooks/02_prune_and_measure.ipynb) · часть проекта [LinguaPrune](https://github.com/s0nakh/linguaprune)*
#
# **Вопрос:** если убрать экспертов, которые редко нужны казахскому и русскому, станет ли модель
# хуже понимать казахский и русский?
#
# **Аналогия.** В компании 64 сотрудника на этаж. Мы увольняем часть и смотрим, справляется ли
# офис с казахскими и русскими клиентами. Чтобы проверка была честной, сравниваем 4 способа
# выбрать, кого оставить:
#
# | Способ | Кого оставляем | Зачем |
# |---|---|---|
# | `freq_kkru` | тех, кого роутер чаще всего зовёт на kk и ru | **наш метод** |
# | `weighted_kkru` | то же, но учитываем, *с каким весом* зовёт | улучшенный наш метод |
# | `freq_en` | тех, кого чаще зовут на английском | контроль: важен ли язык калибровки |
# | `random` (×5) | случайных, пять разных жеребьёвок | нижняя граница: если наш метод не лучше — он не работает |
#
# **Как меряем качество** — на Belebele (тексты на kk, ru и en с одинаковым смыслом):
# - **перплексия** — насколько модель «удивлена» текстом. Меньше = лучше понимает язык.
#   Это **главная** метрика: она считается по ~40 тыс. токенов на язык и поэтому точная;
# - **точность** на вопросах к тексту (4 варианта ответа; угадывание = 25%). На 150 вопросах
#   её погрешность около ±8 п.п., поэтому она только для общей картины.
#
# **Запуск:** GPU T4 x2, Internet On, затем **Save Version → Save & Run All (Commit)** —
# прогон идёт ~2–2.5 часа, вкладку можно закрыть. Результаты — во вкладке Output → `results`.

# %%
import json
import platform
import time
from pathlib import Path

import datasets
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import transformers
from datasets import load_dataset
import torch.nn.functional as F
from huggingface_hub import HfApi
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "allenai/OLMoE-1B-7B-0125"
MODEL_DTYPE = torch.float16  # у T4 нет аппаратного bf16
WIKI_DATASET = "wikimedia/wikipedia"
CALIBRATION_LANGUAGES = {"kk": "20231101.kk", "ru": "20231101.ru", "en": "20231101.en"}
EVAL_DATASET = "facebook/belebele"
EVAL_LANGUAGES = {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}

CALIBRATION_TOKENS = 100_000  # токенов Википедии на язык для выбора экспертов
MAX_TOKENS_PER_DOC = 512
MIN_CHARS_PER_DOC = 500
SHUFFLE_BUFFER = 10_000
KEEP_COUNTS = (48, 38, 32, 24)  # сколько экспертов из 64 оставляем в каждом слое; 38 ≈ результат ноутбука 01
RANDOM_SEEDS = (0, 1, 2, 3, 4)  # 5 независимых жеребьёвок на каждый K: виден разброс, а не одна удача
METHODS = ("freq_kkru", "weighted_kkru", "freq_en", *(f"random_s{n}" for n in RANDOM_SEEDS))
N_EVAL = 150  # текстов Belebele на язык (одни и те же отрывки во всех языках)
SEED = 0

KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR = (KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")) / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("GPU не найдена. На Kaggle включи Settings → Accelerator → GPU T4 x2.")

torch.manual_seed(SEED)
VERSIONS = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
}
print(VERSIONS, f"| GPU x{torch.cuda.device_count()}")

# %% [markdown]
# ## 1. Загружаем модель
#
# Тот же OLMoE, что в ноутбуке 01. В каждом из 16 слоёв роутер выбирает 8 экспертов из 64.

# %%
def load_model(model_id: str, revision: str):
    """Грузим ровно ту версию весов, номер которой запишем в отчёт, и проверяем, что всё на GPU."""
    loaded = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=MODEL_DTYPE, device_map="auto")
    if loaded.dtype != MODEL_DTYPE:
        raise RuntimeError(f"Модель загрузилась в {loaded.dtype}, а нужен {MODEL_DTYPE}.")
    offloaded = {k: v for k, v in getattr(loaded, "hf_device_map", {}).items() if str(v) in ("cpu", "disk")}
    if offloaded:
        raise RuntimeError(f"Часть модели не поместилась на GPU: {offloaded}")
    return loaded


def find_routers(model) -> list[torch.nn.Module]:
    routers = [m for name, m in model.named_modules() if name.endswith("mlp.gate")]
    if not routers:
        raise RuntimeError("Не найдено ни одного роутера 'mlp.gate'.")
    for router in routers:
        if not all(hasattr(router, a) for a in ("weight", "top_k", "norm_topk_prob")):
            raise RuntimeError(f"Неизвестный тип роутера {type(router).__name__}: нужен transformers 5.x.")
    return routers


MODEL_REVISION = HfApi().model_info(MODEL_ID).sha  # фиксируем версию весов
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = load_model(MODEL_ID, MODEL_REVISION)
model.eval()
ROUTERS = find_routers(model)
NUM_LAYERS = len(ROUTERS)
NUM_EXPERTS = ROUTERS[0].weight.shape[0]
TOP_K = ROUTERS[0].top_k
INPUT_DEVICE = model.get_input_embeddings().weight.device
print(f"Слоёв: {NUM_LAYERS}, экспертов: {NUM_EXPERTS}, активных: {TOP_K}, ревизия: {MODEL_REVISION}")

# %% [markdown]
# ## 2. Как «уволить» эксперта, не трогая файлы модели
#
# Роутер считает для каждого эксперта число-«оценку» и отдаёт токен 8 экспертам с лучшими оценками.
# Мы перехватываем роутер и уволенным экспертам ставим оценку **минус бесконечность**:
# их больше никогда не выберут, а веса оставшихся пересчитаются, как будто уволенных и не было.
# Математически это **то же самое**, что физически удалить экспертов из модели —
# только проще и обратимо (память мы сэкономим позже, когда метод докажет, что работает).
#
# Тот же перехватчик в режиме «запись» считает, как часто и с каким весом выбирают каждого эксперта.

# %%
def masked_routing(router, logits: torch.Tensor, mask: torch.Tensor):
    """Повторяет логику роутера OLMoE (transformers 5.x), но с маской уволенных экспертов."""
    masked_logits = logits + mask
    probs = torch.softmax(masked_logits, dtype=torch.float, dim=-1)
    top_value, top_index = torch.topk(probs, router.top_k, dim=-1)
    if router.norm_topk_prob:
        top_value = top_value / top_value.sum(dim=-1, keepdim=True)
    return masked_logits, top_value.to(logits.dtype), top_index


class RouterController:
    """Перехватывает все роутеры: увольняет экспертов по маскам и (по желанию) ведёт статистику."""

    def __init__(self, routers: list[torch.nn.Module]):
        self.masks: list[torch.Tensor | None] = [None] * len(routers)
        self.recording = False
        self.reset_stats()
        self.handles = [r.register_forward_hook(self._make_hook(i)) for i, r in enumerate(routers)]

    def reset_stats(self) -> None:
        self.counts = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.int64)
        self.weighted = np.zeros((NUM_LAYERS, NUM_EXPERTS), dtype=np.float64)

    def _make_hook(self, layer: int):
        def hook(router, _inputs, output):
            if not (isinstance(output, tuple) and len(output) == 3):
                raise RuntimeError("Роутер вернул не (logits, scores, indices) — другая версия transformers.")
            if self.masks[layer] is not None:
                output = masked_routing(router, output[0], self.masks[layer])
            if self.recording:
                self._record(layer, output)
            return output
        return hook

    def _record(self, layer: int, output) -> None:
        _, top_value, top_index = output
        index = top_index[1:].reshape(-1)  # первый токен не считаем («сток внимания»)
        weight = top_value[1:].float().reshape(-1)
        self.counts[layer] += torch.bincount(index, minlength=NUM_EXPERTS).cpu().numpy()
        weighted = torch.zeros(NUM_EXPERTS, device=index.device).index_add_(0, index, weight)
        self.weighted[layer] += weighted.cpu().numpy()

    def set_kept(self, kept: np.ndarray | None) -> None:
        """kept: [слои, K] — номера оставленных экспертов; None — вернуть всех."""
        if kept is None:
            self.masks = [None] * NUM_LAYERS
            return
        self.masks = [build_mask(kept[layer], router) for layer, router in enumerate(ROUTERS)]

    def remove(self) -> None:
        for handle in self.handles:
            handle.remove()


def build_mask(kept_layer: np.ndarray, router) -> torch.Tensor:
    mask = torch.full((NUM_EXPERTS,), float("-inf"), device=router.weight.device)
    mask[torch.as_tensor(kept_layer, device=router.weight.device)] = 0.0
    return mask.to(router.weight.dtype)


controller = RouterController(ROUTERS)

# %% [markdown]
# ### Самопроверка увольнения (четыре проверки)
#
# 1. Если «уволить» никого (маска из нулей), ответы модели должны совпасть с исходными.
# 2. Если уволить половину, ответы должны измениться (иначе перехват не работает).
# 3. Маска должна давать **ровно то же**, что настоящее удаление: берём роутер, физически
#    вырезаем из него строки уволенных экспертов и сравниваем выбор и веса с нашей маской.
# 4. В настоящем прогоне модели ни один слой не должен выбрать уволенного эксперта.
#
# Плюс защита от переполнения fp16: если где-то появится NaN, ноутбук остановится, а не выдаст мусор.

# %%
SELF_TEST_TEXTS = (
    "Қазақстан — Орталық Азиядағы мемлекет.",
    "Москва — столица России.",
    "The quick brown fox jumps over the lazy dog.",
)


@torch.inference_mode()
def model_logits(text: str) -> torch.Tensor:
    ids = tokenizer(text, return_tensors="pt").input_ids.to(INPUT_DEVICE)
    logits = model(ids, use_cache=False).logits.float()
    if not torch.isfinite(logits).all():
        raise FloatingPointError("NaN/inf в выходе модели: fp16 переполнился.")
    return logits


def max_diff_vs_reference(kept: np.ndarray) -> float:
    diffs = []
    for text in SELF_TEST_TEXTS:
        controller.set_kept(None)
        reference = model_logits(text)
        controller.set_kept(kept)
        diffs = [*diffs, (model_logits(text) - reference).abs().max().item()]
    controller.set_kept(None)
    return max(diffs)


@torch.inference_mode()
def mask_vs_physical_removal(router, kept_layer: np.ndarray, num_tokens: int = 256) -> tuple[float, float]:
    """Сравниваем маску с роутером, из которого уволенные эксперты вырезаны по-настоящему.

    Оба варианта считаем из ОДНИХ и тех же оценок роутера: вырезать строки весов роутера —
    то же самое, что взять только столбцы оставленных экспертов из его оценок. Так сравнение
    не зависит от шума округления fp16 в разных умножениях матриц.
    Возвращает (доля токенов с тем же выбором экспертов, макс. разница весов).
    """
    device, dtype = router.weight.device, router.weight.dtype
    hidden = torch.randn(num_tokens, router.weight.shape[1], device=device, dtype=dtype)
    logits = F.linear(hidden, router.weight)
    _, masked_w, masked_i = masked_routing(router, logits, build_mask(kept_layer, router))
    kept_t = torch.as_tensor(kept_layer, device=device)
    probs = torch.softmax(logits[:, kept_t], dtype=torch.float, dim=-1)
    phys_w, phys_local = torch.topk(probs, router.top_k, dim=-1)
    if router.norm_topk_prob:
        phys_w = phys_w / phys_w.sum(dim=-1, keepdim=True)
    same_rows = (kept_t[phys_local].sort(-1).values == masked_i.sort(-1).values).all(-1)
    agreement = same_rows.float().mean().item()
    if not same_rows.any():
        return agreement, float("inf")
    phys_w = phys_w.to(masked_w.dtype).float()  # обе стороны в одной точности, как в настоящем роутере
    weight_diff = (phys_w.sort(-1).values - masked_w.float().sort(-1).values).abs()[same_rows].max().item()
    return agreement, weight_diff


half = np.tile(np.arange(NUM_EXPERTS // 2), (NUM_LAYERS, 1))
diff_same = max_diff_vs_reference(np.tile(np.arange(NUM_EXPERTS), (NUM_LAYERS, 1)))
diff_changed = max_diff_vs_reference(half)
physical = [mask_vs_physical_removal(r, half[i]) for i, r in enumerate(ROUTERS)]
physical_agreement = min(a for a, _ in physical)
physical_weight_diff = max(d for _, d in physical)
physical_ok = physical_agreement >= 0.999 and physical_weight_diff <= 1e-3


def live_hook_respects_mask(kept: np.ndarray) -> bool:
    """Прогоняем настоящий текст с маской и смотрим, кого роутеры реально выбрали в каждом слое."""
    controller.set_kept(kept)
    controller.reset_stats()
    controller.recording = True
    try:
        for text in SELF_TEST_TEXTS:
            model_logits(text)
    finally:
        controller.recording = False
        controller.set_kept(None)
    dropped = np.ones((NUM_LAYERS, NUM_EXPERTS), dtype=bool)
    np.put_along_axis(dropped, kept, False, axis=1)
    ok = controller.counts[dropped].sum() == 0 and (controller.counts.sum(axis=1) > 0).all()
    controller.reset_stats()
    return bool(ok)


live_ok = live_hook_respects_mask(half)
print(f"1) Никого не уволили: разница {diff_same:.4f} (должна быть ≈0)")
print(f"2) Уволили половину:  разница {diff_changed:.2f} (должна быть заметно больше 0)")
print(f"3) Маска = настоящее удаление: совпадение выбора {physical_agreement:.4f} (≥0.999), "
      f"разница весов {physical_weight_diff:.2e} (≤1e-3) → {physical_ok}")
print(f"4) В живом прогоне уволенных экспертов не выбрал ни один слой: {live_ok}")
checks_passed = (diff_same <= 1e-2) and (diff_changed >= 1e-1) and physical_ok and live_ok  # NaN не пройдёт
if not checks_passed:
    raise RuntimeError("Самопроверка увольнения не прошла — результатам верить нельзя.")

# %% [markdown]
# ## 3. Калибровка: кого роутер зовёт на казахском, русском и английском
#
# Как в ноутбуке 01: прогоняем 100 тыс. токенов Википедии на каждом языке и считаем выборы.
# Качество потом меряем на **других** текстах (Belebele) — иначе проверка была бы нечестной.

# %%
def iter_wiki_texts(wiki_config: str):
    try:
        stream = load_dataset(WIKI_DATASET, wiki_config, split="train", streaming=True)
    except Exception as error:
        raise RuntimeError(f"Не удалось открыть {WIKI_DATASET}/{wiki_config}. Internet включён?") from error
    for row in stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER):
        text = row["text"].strip()
        if len(text) >= MIN_CHARS_PER_DOC:
            yield text


@torch.inference_mode()
def calibrate(wiki_config: str) -> dict[str, np.ndarray]:
    controller.reset_stats()
    controller.recording = True
    tokens_seen = 0
    try:
        for text in iter_wiki_texts(wiki_config):
            ids = tokenizer(text, return_tensors="pt", truncation=True, max_length=MAX_TOKENS_PER_DOC).input_ids
            if ids.shape[1] < 2:
                continue
            model(ids.to(INPUT_DEVICE), use_cache=False)
            tokens_seen += ids.shape[1] - 1
            if tokens_seen >= CALIBRATION_TOKENS:
                break
    finally:
        controller.recording = False
    return {"counts": controller.counts.copy(), "weighted": controller.weighted.copy()}


calibration = {}
for lang, wiki_config in CALIBRATION_LANGUAGES.items():
    started = time.time()
    calibration = {**calibration, lang: calibrate(wiki_config)}
    print(f"калибровка {lang}: {time.time() - started:.0f} с")

# %% [markdown]
# ## 4. Четыре способа выбрать, кого оставить
#
# Для каждого способа у каждого эксперта в каждом слое есть «оценка полезности».
# Оставляем K экспертов с лучшими оценками — одинаковое K для всех способов, чтобы сравнение было честным.

# %%
def row_share(x: np.ndarray) -> np.ndarray:
    return x / x.sum(axis=1, keepdims=True)


def build_scores() -> dict[str, np.ndarray]:
    kk, ru, en = calibration["kk"], calibration["ru"], calibration["en"]
    return {
        "freq_kkru": (row_share(kk["counts"]) + row_share(ru["counts"])) / 2,
        "weighted_kkru": (row_share(kk["weighted"]) + row_share(ru["weighted"])) / 2,
        "freq_en": row_share(en["counts"]),
    }


def random_score(seed_index: int, keep: int) -> np.ndarray:
    """Своя жеребьёвка для каждой пары (номер, K), чтобы наборы разных K не были вложены друг в друга."""
    return np.random.default_rng([SEED, 100 + seed_index, keep]).random((NUM_LAYERS, NUM_EXPERTS))


def score_for(method: str, keep: int) -> np.ndarray:
    if method.startswith("random_s"):
        return random_score(int(method.removeprefix("random_s")), keep)
    return SCORES[method]


def top_experts(score: np.ndarray, keep: int) -> np.ndarray:
    return np.sort(np.argsort(-score, axis=1, kind="stable")[:, :keep], axis=1)


SCORES = build_scores()
overlap = np.mean([len(np.intersect1d(a, b)) for a, b in zip(top_experts(SCORES["freq_kkru"], 38),
                                                                top_experts(SCORES["freq_en"], 38))])
print(f"При K=38 наборы 'kk+ru' и 'en' совпадают в среднем на {overlap:.1f} эксперта из 38")

# %% [markdown]
# ## 5. Экзамен: Belebele
#
# Берём одни и те же 150 отрывков на трёх языках. Для каждого:
# - считаем перплексию на самом отрывке;
# - задаём вопрос с 4 вариантами и смотрим, какой букве (A/B/C/D) модель даёт больше вероятности.

# %%
ANSWER_LETTERS = (" A", " B", " C", " D")


def load_eval_rows() -> dict[str, pd.DataFrame]:
    frames = {lang: load_dataset(EVAL_DATASET, cfg, split="test").to_pandas() for lang, cfg in EVAL_LANGUAGES.items()}
    keys = frames["en"].drop_duplicates("link")[["link", "question_number"]]  # один вопрос на отрывок
    keys = keys.sample(n=min(N_EVAL, len(keys)), random_state=SEED)
    rows = {lang: keys.merge(df, on=["link", "question_number"], how="left", validate="one_to_one")
            for lang, df in frames.items()}
    missing = {lang: int(df.isna().any(axis=1).sum()) for lang, df in rows.items()}
    if any(missing.values()):
        raise RuntimeError(f"В Belebele не нашлись пары для некоторых вопросов: {missing}")
    return rows


def encode(text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def token_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    """Номера токенов, которые покрывают символы [char_start, char_end) текста."""
    first = next(i for i, (_, end) in enumerate(offsets) if end > char_start)
    last = next((i for i, (start, _) in enumerate(offsets) if start >= char_end), len(offsets))
    return first, last


def build_example(row) -> dict:
    """Кодируем весь промпт одной строкой — так токены на стыках такие же, как в обычном тексте."""
    prefix, passage = "Passage: ", row["flores_passage"].strip()
    options = "".join(f"\n{letter.strip()}. {row[f'mc_answer{i + 1}']}" for i, letter in enumerate(ANSWER_LETTERS))
    text = f"{prefix}{passage}\nQuestion: {row['question'].strip()}{options}\nAnswer:"
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = token_span(encoded["offset_mapping"], len(prefix), len(prefix) + len(passage))
    if start < 1 or end <= start:
        raise RuntimeError("Не удалось найти отрывок среди токенов.")
    return {
        "ids": torch.tensor([encoded["input_ids"]]),
        "passage_span": (start, end),
        "label": int(row["correct_answer_num"]) - 1,
    }


ANSWER_IDS = [encode(letter) for letter in ANSWER_LETTERS]
if any(len(ids) != 1 for ids in ANSWER_IDS):
    raise RuntimeError("Буквы ответов должны быть одним токеном каждая.")
ANSWER_IDS = [ids[0] for ids in ANSWER_IDS]

EVAL_ROWS = load_eval_rows()
EXAMPLES = {lang: [build_example(row) for _, row in df.iterrows()] for lang, df in EVAL_ROWS.items()}
print({lang: len(ex) for lang, ex in EXAMPLES.items()}, "примеров на язык")

# %%
@torch.inference_mode()
def evaluate_language(examples: list[dict]) -> dict[str, float]:
    total_nll, total_tokens, correct = 0.0, 0, 0
    picked = np.zeros(len(ANSWER_IDS), dtype=np.int64)
    for ex in examples:
        ids = ex["ids"].to(INPUT_DEVICE)
        logits = model(ids, use_cache=False).logits[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("NaN/inf в выходе модели: fp16 переполнился.")
        start, end = ex["passage_span"]
        log_probs = torch.log_softmax(logits[start - 1:end - 1].float(), dim=-1)
        targets = ids[0, start:end].to(log_probs.device)
        total_nll += -log_probs.gather(1, targets[:, None]).sum().item()
        total_tokens += end - start
        choice = int(logits[-1, ANSWER_IDS].argmax().item())
        picked[choice] += 1
        correct += int(choice == ex["label"])
    return {
        "perplexity": float(np.exp(total_nll / total_tokens)),
        "accuracy": correct / len(examples),
        "picked_ABCD": "/".join(str(n) for n in picked),  # если модель всегда жмёт одну букву — это видно
    }

# %% [markdown]
# ## 6. Главный прогон
#
# Сначала исходная модель (все 64 эксперта), затем 4 способа × 4 размера команды = 16 вариантов.
# После каждого варианта результаты дописываются в `results/results.csv` —
# если сессия оборвётся, сделанное не пропадёт.

# %%
def run_config(method: str, keep: int) -> list[dict]:
    kept = None if method == "base" else top_experts(score_for(method, keep), keep)
    controller.set_kept(kept)
    rows = []
    for lang, examples in EXAMPLES.items():
        metrics = evaluate_language(examples)
        rows = [*rows, {"method": method, "keep": keep, "lang": lang, **metrics}]
    controller.set_kept(None)
    return rows


configs = [("base", NUM_EXPERTS)] + [(m, k) for k in KEEP_COUNTS for m in METHODS]
results: list[dict] = []
for number, (method, keep) in enumerate(configs, start=1):
    started = time.time()
    results = [*results, *run_config(method, keep)]
    pd.DataFrame(results).to_csv(OUTPUT_DIR / "results.csv", index=False)
    took = time.time() - started
    print(f"[{number}/{len(configs)}] {method:>14} K={keep:<2} | {took:.0f} с | "
          f"осталось ~{took * (len(configs) - number) / 60:.0f} мин")

# %% [markdown]
# ## 7. Результаты
#
# **Как читать таблицу:** число — во сколько раз выросла перплексия по сравнению с исходной моделью.
# `1.00` = ничего не потеряли, `1.20` = модель «удивляется» тексту на 20% больше.

# %%
df = pd.DataFrame(results)
if df.duplicated(["method", "keep", "lang"]).any():
    raise RuntimeError("Дубликаты в результатах — таблицы были бы усреднены молча.")
base = df[df.method == "base"].set_index("lang")
if not np.isfinite(base.perplexity).all():
    raise RuntimeError("Перплексия исходной модели не число — что-то сломалось.")
df["ppl_ratio"] = df.apply(lambda r: r.perplexity / base.loc[r.lang, "perplexity"], axis=1)
print("Исходная модель:")
print(base[["perplexity", "accuracy", "picked_ABCD"]].round(3))
df["family"] = df.method.str.replace(r"_s\d+$", "", regex=True)  # random_s0/1/2 → random
random_part = df[df.family == "random"]
spread = random_part.groupby(["keep", "lang"]).ppl_ratio.agg(["min", "max"]).unstack("lang")
print("\nРазброс случайного выбора (min–max по 5 жеребьёвкам):")
print(spread.round(3))
ratio_table = df[df.method != "base"].pivot_table(index=["keep", "method"], columns="lang", values="ppl_ratio")
print("\nРост перплексии (меньше = лучше):")
print(ratio_table[["kk", "ru", "en"]].round(3))
accuracy_table = df[df.method != "base"].pivot_table(index=["keep", "method"], columns="lang", values="accuracy")
print("\nТочность на вопросах (угадывание = 0.25):")
print(accuracy_table[["kk", "ru", "en"]].round(3))

# %%
fig, axes = plt.subplots(1, 3, figsize=(16, 4), sharey=True)
for ax, lang in zip(axes, ("kk", "ru", "en")):
    for method in METHODS:
        part = df[(df.method == method) & (df.lang == lang)].sort_values("keep")
        is_random = method.startswith("random")
        ax.plot(part.keep, part.ppl_ratio, marker="o", color="gray" if is_random else None,
                alpha=0.5 if is_random else 1.0, label="random (×5)" if method == "random_s0" else
                (None if is_random else method))
    ax.axhline(1.0, color="gray", ls="--", lw=0.8)
    ax.set_yscale("log")  # логарифмическая шкала: случайное увольнение может увеличить перплексию в тысячи раз
    ax.set_title(f"{lang}: рост перплексии")
    ax.set_xlabel("оставлено экспертов из 64")
    ax.invert_xaxis()
axes[0].set_ylabel("перплексия / исходная")
axes[0].legend()
fig.savefig(OUTPUT_DIR / "perplexity_vs_experts.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Сохраняем
#
# **Что мы надеемся увидеть:** на kk и ru линии `freq_kkru`/`weighted_kkru` заметно ниже, чем `random`
# и `freq_en`. На en — наоборот (английским мы жертвуем сознательно).
# **Если `freq_kkru` попадает в серый «коридор» случайных** — частота выбора не годится как критерий
# (гипотеза: тогда поможет REAP, который учитывает ещё и силу вклада эксперта).
#
# **Ограничения:** калибровка на Википедии, проверка на Belebele — разные тексты (это честно, но домены
# различаются); K одинаковое во всех слоях, хотя слои используют экспертов по-разному.

# %%
controller.remove()
summary = {
    "model": MODEL_ID,
    "model_revision": MODEL_REVISION,
    "dtype": str(MODEL_DTYPE),
    "versions": VERSIONS,
    "seed": SEED,
    "calibration": {"dataset": WIKI_DATASET, "configs": CALIBRATION_LANGUAGES, "tokens": CALIBRATION_TOKENS},
    "eval": {"dataset": EVAL_DATASET, "configs": EVAL_LANGUAGES, "n_per_language": N_EVAL},
    "self_test": {"no_drop_max_diff": diff_same, "half_drop_max_diff": diff_changed,
                  "mask_vs_physical_agreement": physical_agreement,
                  "mask_vs_physical_weight_diff": physical_weight_diff, "live_hook_respects_mask": live_ok},
    "kk_ru_vs_en_overlap_at_38": float(overlap),
    "base": base[["perplexity", "accuracy", "picked_ABCD"]].to_dict(orient="index"),
    "ppl_ratio": {f"{m}@{k}": v for (k, m), v in ratio_table.to_dict(orient="index").items()},
    "accuracy": {f"{m}@{k}": v for (k, m), v in accuracy_table.to_dict(orient="index").items()},
}
(OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))
np.savez_compressed(OUTPUT_DIR / "calibration.npz",
                    **{f"{lang}_{k}": v for lang, stats in calibration.items() for k, v in stats.items()})
print("Готово:", sorted(p.name for p in OUTPUT_DIR.iterdir()))
