# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 02 · Remove experts and check whether the model got worse
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# *[Русская версия](https://github.com/s0nakh/linguaprune/blob/main/notebooks/ru/02_prune_and_measure.ipynb) · part of [LinguaPrune](https://github.com/s0nakh/linguaprune)*
#
# **Question:** if we remove the experts that Kazakh and Russian rarely need, does the model get
# worse at Kazakh and Russian?
#
# **Analogy.** A company has 64 employees per floor. We let some of them go and check whether the
# office still serves Kazakh- and Russian-speaking clients well. To make the test fair, we compare
# four ways of choosing who stays:
#
# | Method | Who stays | Why |
# |---|---|---|
# | `freq_kkru` | experts the router picks most often for kk and ru | **our method** |
# | `weighted_kkru` | same, but weighted by *how strongly* the router picks them | improved variant |
# | `freq_en` | experts picked most often for English | control: does the calibration language matter? |
# | `random` (×5) | random experts, five independent draws | lower bound: if our method is not better, it does not work |
#
# **How quality is measured** — on Belebele (the same passages in kk, ru and en):
# - **perplexity** — how "surprised" the model is by the text. Lower = understands the language better.
#   This is the **primary** metric: it uses ~40k tokens per language, so it is precise;
# - **accuracy** on reading-comprehension questions (4 options; chance = 25%). With 150 questions
#   its error is about ±8 points, so it is only a rough signal.
#
# **Run:** GPU T4 x2, Internet On, then **Save Version → Save & Run All (Commit)** —
# takes ~2–2.5 hours, the tab can be closed. Results are in the Output tab → `results`.

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
MODEL_DTYPE = torch.float16  # T4 has no hardware bf16
WIKI_DATASET = "wikimedia/wikipedia"
CALIBRATION_LANGUAGES = {"kk": "20231101.kk", "ru": "20231101.ru", "en": "20231101.en"}
EVAL_DATASET = "facebook/belebele"
EVAL_LANGUAGES = {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}

CALIBRATION_TOKENS = 100_000  # Wikipedia tokens per language used to choose experts
MAX_TOKENS_PER_DOC = 512
MIN_CHARS_PER_DOC = 500
SHUFFLE_BUFFER = 10_000
KEEP_COUNTS = (48, 38, 32, 24)  # experts kept per layer (of 64); 38 ≈ the notebook 01 result
RANDOM_SEEDS = (0, 1, 2, 3, 4)  # 5 independent draws per K: shows the spread, not one lucky draw
METHODS = ("freq_kkru", "weighted_kkru", "freq_en", *(f"random_s{n}" for n in RANDOM_SEEDS))
N_EVAL = 150  # Belebele passages per language (the same passages in every language)
SEED = 0

KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR = (KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")) / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. On Kaggle, enable Settings → Accelerator → GPU T4 x2.")

torch.manual_seed(SEED)
VERSIONS = {
    "python": platform.python_version(),
    "torch": torch.__version__,
    "transformers": transformers.__version__,
    "datasets": datasets.__version__,
}
print(VERSIONS, f"| GPU x{torch.cuda.device_count()}")

# %% [markdown]
# ## 1. Load the model
#
# The same OLMoE as in notebook 01. In each of the 16 layers the router picks 8 of 64 experts.

# %%
def load_model(model_id: str, revision: str):
    """Load exactly the weight revision we record in the report, and check that it is all on GPU."""
    loaded = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=MODEL_DTYPE, device_map="auto")
    if loaded.dtype != MODEL_DTYPE:
        raise RuntimeError(f"Model loaded as {loaded.dtype}, expected {MODEL_DTYPE}.")
    offloaded = {k: v for k, v in getattr(loaded, "hf_device_map", {}).items() if str(v) in ("cpu", "disk")}
    if offloaded:
        raise RuntimeError(f"Part of the model did not fit on GPU: {offloaded}")
    return loaded


def find_routers(model) -> list[torch.nn.Module]:
    routers = [m for name, m in model.named_modules() if name.endswith("mlp.gate")]
    if not routers:
        raise RuntimeError("No 'mlp.gate' routers found.")
    for router in routers:
        if not all(hasattr(router, a) for a in ("weight", "top_k", "norm_topk_prob")):
            raise RuntimeError(f"Unknown router type {type(router).__name__}: transformers 5.x is required.")
    return routers


MODEL_REVISION = HfApi().model_info(MODEL_ID).sha  # pin the weight revision
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = load_model(MODEL_ID, MODEL_REVISION)
model.eval()
ROUTERS = find_routers(model)
NUM_LAYERS = len(ROUTERS)
NUM_EXPERTS = ROUTERS[0].weight.shape[0]
TOP_K = ROUTERS[0].top_k
INPUT_DEVICE = model.get_input_embeddings().weight.device
print(f"Layers: {NUM_LAYERS}, experts: {NUM_EXPERTS}, active: {TOP_K}, revision: {MODEL_REVISION}")

# %% [markdown]
# ## 2. How to remove an expert without touching the model files
#
# The router gives every expert a score and sends the token to the 8 experts with the best scores.
# We intercept the router and give removed experts a score of **minus infinity**: they can never be
# picked again, and the weights of the remaining experts are renormalized as if the removed ones
# never existed. For the forward pass this is mathematically **the same** as physically deleting the
# experts — just simpler and reversible. (It does not save memory yet; that comes once the method
# has proven itself.)
#
# In "recording" mode the same hook counts how often, and with what weight, each expert is picked.

# %%
def masked_routing(router, logits: torch.Tensor, mask: torch.Tensor):
    """Reproduces the OLMoE router logic (transformers 5.x), with a mask for removed experts."""
    masked_logits = logits + mask
    probs = torch.softmax(masked_logits, dtype=torch.float, dim=-1)
    top_value, top_index = torch.topk(probs, router.top_k, dim=-1)
    if router.norm_topk_prob:
        top_value = top_value / top_value.sum(dim=-1, keepdim=True)
    return masked_logits, top_value.to(logits.dtype), top_index


class RouterController:
    """Intercepts every router: removes experts via masks and, optionally, records statistics."""

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
                raise RuntimeError("Router did not return (logits, scores, indices) — different transformers version.")
            if self.masks[layer] is not None:
                output = masked_routing(router, output[0], self.masks[layer])
            if self.recording:
                self._record(layer, output)
            return output
        return hook

    def _record(self, layer: int, output) -> None:
        _, top_value, top_index = output
        index = top_index[1:].reshape(-1)  # skip the first token (attention sink)
        weight = top_value[1:].float().reshape(-1)
        self.counts[layer] += torch.bincount(index, minlength=NUM_EXPERTS).cpu().numpy()
        weighted = torch.zeros(NUM_EXPERTS, device=index.device).index_add_(0, index, weight)
        self.weighted[layer] += weighted.cpu().numpy()

    def set_kept(self, kept: np.ndarray | None) -> None:
        """kept: [layers, K] — indices of the kept experts; None restores all experts."""
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
# ### Removal self-tests (four checks)
#
# 1. Removing nobody (an all-zero mask) must leave the model's outputs unchanged.
# 2. Removing half of the experts must change the outputs (otherwise the hook does nothing).
# 3. The mask must give **exactly the same** result as real removal: we take the router, physically
#    cut out the rows of removed experts, and compare the chosen experts and weights with our mask.
# 4. In a real forward pass, no layer may pick a removed expert.
#
# Plus an fp16 overflow guard: if a NaN appears anywhere, the notebook stops instead of producing garbage.

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
        raise FloatingPointError("NaN/inf in model output: fp16 overflow.")
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
    """Compare the mask with a router whose removed experts are actually cut out.

    Both sides start from the SAME router scores: removing rows from the router weights is the same
    as taking only the kept experts' columns of its scores. This keeps the comparison independent of
    fp16 rounding noise between different matrix multiplications.
    Returns (share of tokens with the same expert choice, max weight difference).
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
    phys_w = phys_w.to(masked_w.dtype).float()  # both sides in the same precision, as in the real router
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
    """Run real text with the mask and check which experts each layer's router actually picked."""
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
print(f"1) Nobody removed: max diff {diff_same:.4f} (should be ≈0)")
print(f"2) Half removed:   max diff {diff_changed:.2f} (should be clearly above 0)")
print(f"3) Mask = real removal: choice agreement {physical_agreement:.4f} (≥0.999), "
      f"weight diff {physical_weight_diff:.2e} (≤1e-3) → {physical_ok}")
print(f"4) Live run: no layer picked a removed expert: {live_ok}")
checks_passed = (diff_same <= 1e-2) and (diff_changed >= 1e-1) and physical_ok and live_ok  # NaN fails this
if not checks_passed:
    raise RuntimeError("Removal self-test failed — the results cannot be trusted.")

# %% [markdown]
# ## 3. Calibration: which experts the router uses for Kazakh, Russian and English
#
# As in notebook 01: run 100k Wikipedia tokens per language and count the picks.
# Quality is then measured on **different** texts (Belebele) — otherwise the test would not be fair.

# %%
def iter_wiki_texts(wiki_config: str):
    try:
        stream = load_dataset(WIKI_DATASET, wiki_config, split="train", streaming=True)
    except Exception as error:
        raise RuntimeError(f"Could not open {WIKI_DATASET}/{wiki_config}. Is Internet enabled?") from error
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
    print(f"calibration {lang}: {time.time() - started:.0f} s")

# %% [markdown]
# ## 4. Four ways to choose who stays
#
# Each method gives every expert in every layer a "usefulness score".
# We keep the K experts with the best scores — the same K for every method, so the comparison is fair.

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
    """A separate draw for every (seed, K) pair, so sets for different K are not nested in each other."""
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
print(f"At K=38 the 'kk+ru' and 'en' sets share {overlap:.1f} of 38 experts on average")

# %% [markdown]
# ## 5. The exam: Belebele
#
# We take the same 150 passages in three languages. For each one we:
# - compute perplexity on the passage itself;
# - ask a 4-option question and check which letter (A/B/C/D) the model gives the highest probability.

# %%
ANSWER_LETTERS = (" A", " B", " C", " D")


def load_eval_rows() -> dict[str, pd.DataFrame]:
    frames = {lang: load_dataset(EVAL_DATASET, cfg, split="test").to_pandas() for lang, cfg in EVAL_LANGUAGES.items()}
    keys = frames["en"].drop_duplicates("link")[["link", "question_number"]]  # one question per passage
    keys = keys.sample(n=min(N_EVAL, len(keys)), random_state=SEED)
    rows = {lang: keys.merge(df, on=["link", "question_number"], how="left", validate="one_to_one")
            for lang, df in frames.items()}
    missing = {lang: int(df.isna().any(axis=1).sum()) for lang, df in rows.items()}
    if any(missing.values()):
        raise RuntimeError(f"Some Belebele questions have no counterpart in every language: {missing}")
    return rows


def encode(text: str) -> list[int]:
    return tokenizer(text, add_special_tokens=False).input_ids


def token_span(offsets: list[tuple[int, int]], char_start: int, char_end: int) -> tuple[int, int]:
    """Indices of the tokens covering characters [char_start, char_end) of the text."""
    first = next(i for i, (_, end) in enumerate(offsets) if end > char_start)
    last = next((i for i, (start, _) in enumerate(offsets) if start >= char_end), len(offsets))
    return first, last


def build_example(row) -> dict:
    """Encode the whole prompt as one string, so tokens at the boundaries match normal text."""
    prefix, passage = "Passage: ", row["flores_passage"].strip()
    options = "".join(f"\n{letter.strip()}. {row[f'mc_answer{i + 1}']}" for i, letter in enumerate(ANSWER_LETTERS))
    text = f"{prefix}{passage}\nQuestion: {row['question'].strip()}{options}\nAnswer:"
    encoded = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    start, end = token_span(encoded["offset_mapping"], len(prefix), len(prefix) + len(passage))
    if start < 1 or end <= start:
        raise RuntimeError("Could not locate the passage among the tokens.")
    return {
        "ids": torch.tensor([encoded["input_ids"]]),
        "passage_span": (start, end),
        "label": int(row["correct_answer_num"]) - 1,
    }


ANSWER_IDS = [encode(letter) for letter in ANSWER_LETTERS]
if any(len(ids) != 1 for ids in ANSWER_IDS):
    raise RuntimeError("Each answer letter must be a single token.")
ANSWER_IDS = [ids[0] for ids in ANSWER_IDS]

EVAL_ROWS = load_eval_rows()
EXAMPLES = {lang: [build_example(row) for _, row in df.iterrows()] for lang, df in EVAL_ROWS.items()}
print({lang: len(ex) for lang, ex in EXAMPLES.items()}, "examples per language")

# %%
@torch.inference_mode()
def evaluate_language(examples: list[dict]) -> dict[str, float]:
    total_nll, total_tokens, correct = 0.0, 0, 0
    picked = np.zeros(len(ANSWER_IDS), dtype=np.int64)
    for ex in examples:
        ids = ex["ids"].to(INPUT_DEVICE)
        logits = model(ids, use_cache=False).logits[0]
        if not torch.isfinite(logits).all():
            raise FloatingPointError("NaN/inf in model output: fp16 overflow.")
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
        "picked_ABCD": "/".join(str(n) for n in picked),  # shows if the model always presses one letter
    }

# %% [markdown]
# ## 6. Main run
#
# First the original model (all 64 experts), then every method × 4 team sizes.
# After each configuration the results are appended to `results/results.csv` —
# if the session dies, finished work is not lost.

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
    print(f"[{number}/{len(configs)}] {method:>14} K={keep:<2} | {took:.0f} s | "
          f"~{took * (len(configs) - number) / 60:.0f} min left")

# %% [markdown]
# ## 7. Results
#
# **How to read the table:** each number is how many times perplexity grew compared with the
# original model. `1.00` = nothing lost, `1.20` = the model is 20% more "surprised" by the text.

# %%
df = pd.DataFrame(results)
if df.duplicated(["method", "keep", "lang"]).any():
    raise RuntimeError("Duplicate rows in results — tables would be averaged silently.")
base = df[df.method == "base"].set_index("lang")
if not np.isfinite(base.perplexity).all():
    raise RuntimeError("Original model perplexity is not a number — something broke.")
df["ppl_ratio"] = df.apply(lambda r: r.perplexity / base.loc[r.lang, "perplexity"], axis=1)
print("Original model:")
print(base[["perplexity", "accuracy", "picked_ABCD"]].round(3))
df["family"] = df.method.str.replace(r"_s\d+$", "", regex=True)  # random_s0/1/2 → random
random_part = df[df.family == "random"]
spread = random_part.groupby(["keep", "lang"]).ppl_ratio.agg(["min", "max"]).unstack("lang")
print("\nSpread of random selection (min–max over 5 draws):")
print(spread.round(3))
ratio_table = df[df.method != "base"].pivot_table(index=["keep", "method"], columns="lang", values="ppl_ratio")
print("\nPerplexity growth (lower = better):")
print(ratio_table[["kk", "ru", "en"]].round(3))
accuracy_table = df[df.method != "base"].pivot_table(index=["keep", "method"], columns="lang", values="accuracy")
print("\nQuestion accuracy (chance = 0.25):")
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
    ax.set_yscale("log")  # log scale: random removal can grow perplexity thousands of times
    ax.set_title(f"{lang}: perplexity growth")
    ax.set_xlabel("experts kept (of 64)")
    ax.invert_xaxis()
axes[0].set_ylabel("perplexity / original")
axes[0].legend()
fig.savefig(OUTPUT_DIR / "perplexity_vs_experts.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 8. Save
#
# **What we hope to see:** for kk and ru, the `freq_kkru`/`weighted_kkru` lines are clearly below
# `random` and `freq_en`. For en it is the opposite (we sacrifice English on purpose).
# **If `freq_kkru` falls inside the gray "corridor" of random draws**, selection frequency is not a
# good criterion (hypothesis: REAP, which also accounts for each expert's contribution, would help).
#
# **Limitations:** calibration on Wikipedia, evaluation on Belebele — different texts (fair, but the
# domains differ); K is the same in every layer, although layers use experts differently.

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
print("Done:", sorted(p.name for p in OUTPUT_DIR.iterdir()))
