# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 01 · Which MoE experts "speak" Kazakh
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# *[Русская версия](https://github.com/s0nakh/linguaprune/blob/main/notebooks/ru/01_expert_profiling.ipynb) · part of [LinguaPrune](https://github.com/s0nakh/linguaprune)*
#
# **Goal:** find out which experts of a Mixture-of-Experts (MoE) model actually work when the model
# reads Kazakh text, and estimate how many experts could be removed.
#
# **Model:** `allenai/OLMoE-1B-7B-0125` (open, Apache 2.0; 64 experts per layer, 8 active per token).
# A small stand-in for large MoE models: same method, but it fits in free Kaggle GPUs.
#
# **Data:** Wikipedia (`wikimedia/wikipedia`, 2023-11-01 snapshot) in Kazakh, Russian, English and
# Chinese — open data only. Chinese Wikipedia mixes Simplified and Traditional characters.
#
# **Run on Kaggle:** Settings → Accelerator → **GPU T4 x2**, Internet → **On**, then
# **Save Version → Save & Run All**. Figures and JSON are written to `/kaggle/working/results`.
#
# **What this notebook does NOT prove.** We count *how often* the router picks each expert.
# Frequency ≠ importance: a rarely used expert can still be critical for rare but necessary tokens.
# So "covering 90% of router picks" ≠ "keeping 90% of quality". Quality is measured in notebook 02.

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
MODEL_DTYPE = torch.float16  # T4 has no hardware bf16; fp16 + NaN/inf checks below
WIKI_DATASET = "wikimedia/wikipedia"
LANGUAGES = {  # language code -> Wikipedia config (fixed snapshot)
    "kk": "20231101.kk",
    "ru": "20231101.ru",
    "en": "20231101.en",
    "zh": "20231101.zh",
}
TARGET_LANGUAGES = ("kk", "ru")  # languages we prune the model for
TOKENS_PER_LANGUAGE = 100_000  # same token budget for every language
MAX_TOKENS_PER_DOC = 512
MIN_CHARS_PER_DOC = 500
SHUFFLE_BUFFER = 10_000
COVERAGE_TARGET = 0.90  # how many experts cover 90% of router picks
SPECIFICITY_MIN_SHARE = 1e-3  # experts below this share in both groups are left blank
SELF_TEST_MIN_AGREEMENT = 0.99  # share of tokens where our hook matches the model's own choice
BOOTSTRAP_ROUNDS = 200
SEED = 0

KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR = (KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")) / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. On Kaggle, enable Settings → Accelerator → GPU T4 x2.")

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
# ## 1. Load the model and find the routers
#
# In every layer the router (`mlp.gate`) decides which experts receive a token:
# it outputs 64 numbers (logits), and the token goes to the 8 experts with the largest ones.
# We attach a forward hook to each router and record those numbers.

# %%
def read_moe_config(config) -> tuple[int, int]:
    """Returns (experts per layer, experts active per token)."""
    num_experts = getattr(config, "num_experts", None) or getattr(config, "n_routed_experts", None)
    top_k = getattr(config, "num_experts_per_tok", None)
    if not num_experts or not top_k:
        raise ValueError(f"Could not read MoE parameters from config {type(config).__name__}")
    return int(num_experts), int(top_k)


def find_router_modules(model) -> list[torch.nn.Module]:
    routers = [module for name, module in model.named_modules() if name.endswith("mlp.gate")]
    if not routers:
        raise RuntimeError("No 'mlp.gate' modules found — is this an MoE model?")
    return routers


def load_model(model_id: str, revision: str):
    """Load exactly the weight revision we record in the report, and check that it is all on GPU."""
    loaded = AutoModelForCausalLM.from_pretrained(model_id, revision=revision, dtype=MODEL_DTYPE, device_map="auto")
    if loaded.dtype != MODEL_DTYPE:
        raise RuntimeError(f"Model loaded as {loaded.dtype}, expected {MODEL_DTYPE}.")
    offloaded = {k: v for k, v in getattr(loaded, "hf_device_map", {}).items() if str(v) in ("cpu", "disk")}
    if offloaded:
        raise RuntimeError(f"Part of the model did not fit on GPU: {offloaded}")
    return loaded


MODEL_REVISION = HfApi().model_info(MODEL_ID).sha  # exact weight revision (commit) on the Hub
tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, revision=MODEL_REVISION)
model = load_model(MODEL_ID, MODEL_REVISION)
model.eval()

NUM_EXPERTS, TOP_K = read_moe_config(model.config)
ROUTERS = find_router_modules(model)
NUM_LAYERS = len(ROUTERS)
INPUT_DEVICE = model.get_input_embeddings().weight.device
print(f"Layers with a router: {NUM_LAYERS}, experts per layer: {NUM_EXPERTS}, active per token: {TOP_K}")
print(f"Model revision: {MODEL_REVISION}")

# %%
def extract_router_logits(output, num_experts: int) -> torch.Tensor:
    """Extracts the [tokens, experts] logits tensor from a router's output."""
    candidates = output if isinstance(output, (tuple, list)) else (output,)
    for tensor in candidates:
        if torch.is_tensor(tensor) and tensor.is_floating_point() and tensor.shape[-1] == num_experts:
            logits = tensor.reshape(-1, num_experts).float()
            if not torch.isfinite(logits).all():
                raise FloatingPointError("NaN/inf in router logits: fp16 overflow.")
            return logits
    raise RuntimeError("No tensor of the expected shape in the router output — check the transformers version.")


def routing_stats(logits: torch.Tensor, num_experts: int, top_k: int) -> tuple[np.ndarray, np.ndarray]:
    """For one layer: how many times each expert was picked, and its total router probability."""
    probs = torch.softmax(logits, dim=-1)
    chosen = probs.topk(top_k, dim=-1).indices.reshape(-1)
    counts = torch.bincount(chosen, minlength=num_experts).cpu().numpy()
    mass = probs.sum(dim=0).cpu().numpy()
    return counts, mass


class RouterRecorder:
    """Keeps the latest output of every router; statistics are computed after each forward pass."""

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
# ### Hook self-test
#
# Before trusting any statistics, check that the hook sees the same decisions as the model itself.
# The model can return its own router logits (`output_router_logits=True`); we compare the chosen
# experts for every token in every layer. If they disagree, we stop here.

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
        raise RuntimeError("The model did not return router logits for every layer.")
    agreements = [
        (topk_sets(hooked[i], TOP_K) == topk_sets(ref.reshape(-1, NUM_EXPERTS), TOP_K)).all(-1).float().mean()
        for i, ref in enumerate(reference)
    ]
    return float(torch.stack(agreements).min())


recorder = RouterRecorder(ROUTERS, NUM_EXPERTS)
agreement = self_test_hooks(recorder, "Қазақстан — Орталық Азиядағы мемлекет. Астанасы — Астана қаласы.")
print(f"Self-test: expert-choice agreement (worst layer) = {agreement:.3f}")
if agreement < SELF_TEST_MIN_AGREEMENT:
    recorder.remove()
    raise RuntimeError("The hook does not see what the model picks — the statistics cannot be trusted.")

# %% [markdown]
# ## 2. Run texts in four languages
#
# Every language gets the same token budget so the comparison is fair.
# The first token of each article is skipped: the model treats it atypically (the "attention sink"),
# and it would skew the statistics.
#
# **Sampling limitation.** Streaming `shuffle` only mixes a buffer of 10,000 articles (plus shard order),
# so we see a slice of Wikipedia, not all of it. That is enough for an estimate; final numbers should
# use a parallel corpus (FLORES, KazParC).

# %%
def iter_wiki_texts(wiki_config: str):
    try:
        stream = load_dataset(WIKI_DATASET, wiki_config, split="train", streaming=True)
    except Exception as error:  # network or config name are the usual causes
        raise RuntimeError(f"Could not open {WIKI_DATASET}/{wiki_config}. Is Internet enabled?") from error
    for row in stream.shuffle(seed=SEED, buffer_size=SHUFFLE_BUFFER):
        text = row["text"].strip()
        if len(text) >= MIN_CHARS_PER_DOC:
            yield text


@torch.inference_mode()
def profile_document(ids: torch.Tensor, recorder: RouterRecorder) -> tuple[np.ndarray, np.ndarray]:
    """Router statistics for one article, first token excluded. Shapes: [layers, experts]."""
    model(ids.to(INPUT_DEVICE), use_cache=False)
    layer_logits = recorder.pop()
    layer_stats = [routing_stats(layer_logits[i][1:], NUM_EXPERTS, TOP_K) for i in range(NUM_LAYERS)]
    return np.stack([c for c, _ in layer_stats]), np.stack([m for _, m in layer_stats])


def profile_language(wiki_config: str, recorder: RouterRecorder) -> dict:
    doc_counts, doc_mass = [], []
    tokens_seen = 0
    for text in iter_wiki_texts(wiki_config):
        budget_left = TOKENS_PER_LANGUAGE - tokens_seen + 1  # +1: the first token is not counted
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
        "doc_counts": np.stack(doc_counts),  # [articles, layers, experts]
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
        print(f"{lang}: {p['tokens']:,} tokens from {p['docs']} articles in {time.time() - started:.0f} s")
finally:
    recorder.remove()

# %% [markdown]
# ## 3. Analysis: how many experts Kazakh actually needs
#
# Two standard ML techniques keep the estimate honest:
# - **held-out split:** experts are chosen on half of the articles and coverage is measured on the
#   other half (like train/test) — otherwise the estimate is inflated;
# - **bootstrap:** resample articles many times and see how much the number of needed experts varies —
#   this gives an interval instead of a single number.

# %%
def selection_share(counts: np.ndarray) -> np.ndarray:
    """Each expert's share of picks within its layer (rows sum to 1)."""
    return counts / counts.sum(axis=1, keepdims=True)


def experts_for_coverage(share: np.ndarray, target: float) -> list[np.ndarray]:
    """For each layer, the smallest set of experts covering `target` of the picks."""
    kept = []
    for layer_share in share:
        order = np.argsort(-layer_share, kind="stable")
        cumulative = np.cumsum(layer_share[order])
        needed = min(int(np.searchsorted(cumulative, target)) + 1, len(order))
        kept.append(np.sort(order[:needed]))
    return kept


def union_for_targets(doc_counts: dict[str, np.ndarray]) -> list[np.ndarray]:
    """Per-layer union of the expert sets of the target languages."""
    per_lang = [experts_for_coverage(selection_share(doc_counts[lang].sum(0)), COVERAGE_TARGET)
                for lang in TARGET_LANGUAGES]
    return [reduce(np.union1d, [kept[layer] for kept in per_lang]) for layer in range(NUM_LAYERS)]


def covered_share(share: np.ndarray, kept: list[np.ndarray]) -> float:
    """Share of picks that land on the kept experts, averaged over layers."""
    return float(np.mean([share[layer, k].sum() for layer, k in enumerate(kept)]))


def specificity(share: dict[str, np.ndarray], lang: str) -> np.ndarray:
    """log2(share in this language / mean share in the others): >0 means picked more often for this language."""
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

print(f"Experts needed for {COVERAGE_TARGET:.0%} of router picks (mean over layers, of {NUM_EXPERTS}):")
for lang, needed in needed_per_layer.items():
    print(f"  {lang}: {needed.mean():.1f}")
print(f"\n{' + '.join(TARGET_LANGUAGES)} together need {kept_for_targets.mean():.1f} of {NUM_EXPERTS}")
print(f"→ estimate: ~{prunable_fraction:.0%} of experts could be removed (hypothesis; quality is tested in notebook 02)")

# %%
def holdout_coverage() -> dict[str, float]:
    """Choose experts on even-numbered articles, measure coverage on odd-numbered ones."""
    train = {lang: doc_counts[lang][0::2] for lang in TARGET_LANGUAGES}
    kept = union_for_targets(train)
    return {lang: covered_share(selection_share(doc_counts[lang][1::2].sum(0)), kept)
            for lang in TARGET_LANGUAGES}


def bootstrap_kept(rounds: int) -> np.ndarray:
    """Mean number of kept experts per layer on bootstrap resamples of the articles."""
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

print(f"Coverage on HELD-OUT articles (target {COVERAGE_TARGET:.0%}):",
      {k: f"{v:.1%}" for k, v in holdout.items()})
print(f"Experts to keep, 95% bootstrap interval: {boot_low:.1f}–{boot_high:.1f} of {NUM_EXPERTS}")
print("Coverage by router probability mass (not by pick count):",
      {k: f"{v:.1%}" for k, v in mass_coverage.items()})

# %% [markdown]
# ## 4. Figures
#
# Caveat: differences between languages here mix "language" and "topic". Wikipedias in different
# languages cover different subjects (Kazakh Wikipedia has many articles about villages of Kazakhstan),
# so a "Kazakh" expert may really be a geography expert. A parallel corpus will separate the two.

# %%
share_max = max(s.max() for s in shares.values())  # shared color scale so panels are comparable
fig, axes = plt.subplots(1, len(shares), figsize=(5 * len(shares), 4), sharey=True)
for ax, (lang, share) in zip(axes, shares.items()):
    im = ax.imshow(share, aspect="auto", cmap="viridis", vmin=0, vmax=share_max)
    ax.set_title(f"{lang}: share of router picks")
    ax.set_xlabel("expert")
axes[0].set_ylabel("layer")
fig.colorbar(im, ax=axes, shrink=0.8)
fig.savefig(OUTPUT_DIR / "expert_share_by_language.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
kk_specificity = specificity(shares, "kk")
limit = np.nanmax(np.abs(kk_specificity))
fig, ax = plt.subplots(figsize=(12, 4))
im = ax.imshow(kk_specificity, aspect="auto", cmap="RdBu_r", vmin=-limit, vmax=limit)
ax.set_title("kk vs. mean of ru/en/zh, log2: red = picked more for Kazakh, blue = less\n"
             "(language and article topic are mixed; blank = rarely used experts)")
ax.set_xlabel("expert")
ax.set_ylabel("layer")
fig.colorbar(im, ax=ax)
fig.savefig(OUTPUT_DIR / "kk_specificity.png", dpi=150, bbox_inches="tight")
plt.show()

# %%
fig, ax = plt.subplots(figsize=(10, 4))
for lang, needed in needed_per_layer.items():
    ax.plot(needed, marker="o", label=lang)
ax.plot(kept_for_targets, "k--", label=" + ".join(TARGET_LANGUAGES) + " (union)")
ax.axhline(NUM_EXPERTS, color="gray", lw=0.8)
ax.set_title(f"Experts needed for {COVERAGE_TARGET:.0%} of router picks")
ax.set_xlabel("layer")
ax.set_ylabel("experts")
ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1))  # legend outside so it does not cover lines
fig.savefig(OUTPUT_DIR / "experts_needed_per_layer.png", dpi=150, bbox_inches="tight")
plt.show()

# %% [markdown]
# ## 5. Save results (used by notebook 02: pruning and quality checks)

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
