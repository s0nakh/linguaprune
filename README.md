<div align="center">

<img src="assets/banner.png" alt="LinguaPrune" width="720">

**Language-aware expert pruning for Mixture-of-Experts LLMs.**
Keep the experts your languages actually use, drop the rest, then quantize.
First target: Kazakh 🇰🇿 + Russian.

<p>
<a href="LICENSE"><img alt="License" src="https://img.shields.io/badge/license-Apache--2.0-A020F0?style=for-the-badge&labelColor=0D1117&logo=apache&logoColor=white"></a>
<a href="https://github.com/s0nakh/linguaprune/stargazers"><img alt="Stars" src="https://img.shields.io/github/stars/s0nakh/linguaprune?style=for-the-badge&labelColor=0D1117&logo=github&color=A020F0"></a>
<a href="https://github.com/s0nakh/linguaprune/commits/main"><img alt="Last commit" src="https://img.shields.io/github/last-commit/s0nakh/linguaprune?style=for-the-badge&labelColor=0D1117&logo=git&logoColor=white&color=39FF14"></a>
<a href="#roadmap"><img alt="Status" src="https://img.shields.io/badge/status-research%20preview-FF8C00?style=for-the-badge&labelColor=0D1117&logo=jupyter&logoColor=white"></a>
</p>
<p>
<a href="https://www.python.org/"><img alt="Python" src="https://img.shields.io/badge/python-3.11%2B-39FF14?style=for-the-badge&labelColor=0D1117&logo=python&logoColor=white"></a>
<a href="https://pytorch.org/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-2.x-EE4C2C?style=for-the-badge&labelColor=0D1117&logo=pytorch&logoColor=white"></a>
<a href="https://github.com/huggingface/transformers"><img alt="Transformers" src="https://img.shields.io/badge/Transformers-5.x-FFD21E?style=for-the-badge&labelColor=0D1117&logo=huggingface&logoColor=white"></a>
<a href="https://www.kaggle.com/"><img alt="Kaggle" src="https://img.shields.io/badge/Kaggle-GPU%20T4%20%C3%972-20BEFF?style=for-the-badge&labelColor=0D1117&logo=kaggle&logoColor=white"></a>
</p>
<p>
<a href="https://huggingface.co/allenai/OLMoE-1B-7B-0125"><img alt="Model" src="https://img.shields.io/badge/model-OLMoE--1B--7B-F0529C?style=for-the-badge&labelColor=0D1117&logo=huggingface&logoColor=white"></a>
<a href="https://huggingface.co/datasets/facebook/belebele"><img alt="Eval" src="https://img.shields.io/badge/eval-Belebele-00C2FF?style=for-the-badge&labelColor=0D1117&logo=meta&logoColor=white"></a>
<a href="#"><img alt="Languages" src="https://img.shields.io/badge/languages-kk%20%C2%B7%20ru-39FF14?style=for-the-badge&labelColor=0D1117&logo=googletranslate&logoColor=white"></a>
<a href="README.ru.md"><img alt="Docs" src="https://img.shields.io/badge/docs-EN%20%2B%20RU-A020F0?style=for-the-badge&labelColor=0D1117&logo=readthedocs&logoColor=white"></a>
</p>

[Русская версия](README.ru.md) · [Related work](docs/prior-art.md)

</div>

---

## Why

Sovereign and multilingual MoE models are huge. A 247B-parameter MoE, for example, needs roughly 494 GB in BF16, or about 7 server GPUs.
But a model serving Kazakh and Russian users rarely calls many of its experts.
LinguaPrune measures **which experts a language really needs** and removes the rest, so the model fits on fewer GPUs.

> **Status (Nov 2026):** we have *measured* what removing experts does to quality — removal is simulated with a router mask.
> The model itself is **not smaller yet**: physically deleting the weights, exporting and 4-bit quantization are the next step.

> We present this as an **empirical study and an open tool**, not as a brand-new method.
> Expert pruning (e.g. REAP) and language-specific pruning already exist; see [related work](docs/prior-art.md).
> Our contribution is Kazakh + Russian: measured, reproducible and honest about quality loss.

## Results so far

**Notebook 01 — router profiling** on [`allenai/OLMoE-1B-7B-0125`](https://huggingface.co/allenai/OLMoE-1B-7B-0125)
(64 experts per layer, 8 active; 100k Wikipedia tokens per language):

| Language | Experts needed for 90% of router picks (of 64) |
|---|---|
| Kazakh (kk) | **22.5** |
| Russian (ru) | 35.9 |
| Chinese (zh) | 39.5 |
| English (en) | 47.8 |

- **kk + ru together need ~37.7 of 64 experts per layer.** That suggests ~41% of experts could go (bootstrap 95% CI for kept experts: 37.3–38.4).
- On **held-out articles**, the chosen experts still cover 96% (kk) and 91% (ru) of router picks.
- ⚠️ This measures how often experts are *chosen*, **not** model quality. Notebook 02 measures quality.

<img src="experiments/01/experts_needed_per_layer.png" alt="Experts needed per layer" width="720">

- **Deep layers need far fewer experts.** In layer 14, Kazakh needs only 12 of 64, while early layers need ~40. This motivates per-layer pruning budgets.
- **Reproducible:** two independent Kaggle runs gave identical numbers and identical per-layer expert sets (model revision `9b0c1aa`).

Raw numbers: [`experiments/01/summary.json`](experiments/01/summary.json).

**Notebook 02 — quality after pruning.** Perplexity is measured on [Belebele](https://huggingface.co/datasets/facebook/belebele): 150 passages with the same content in every language. Values are relative to the unpruned model (1.00× = no loss; lower is better):

| Experts kept (of 64) | kk · ours | kk · random (5 draws) | kk · English-chosen | ru · ours | ru · random | en · ours |
|---|---|---|---|---|---|---|
| 48 | **1.03×** | 2.0–4.6× | 1.55× | **1.09×** | 2.3–5.6× | 1.45× |
| 38 (−41%) | **1.10×** | 16–1668× | 1.86× | **1.27×** | 11–25983× | 2.08× |
| 32 | **1.16×** | 8.5–285× | 27× | **1.51×** | 18–523× | 3.56× |
| 24 | **1.45×** | 393–7544× | 214× | **2.51×** | 981–17520× | 11.2× |

<img src="experiments/02/perplexity_vs_experts.png" alt="Perplexity growth vs. experts kept" width="900">

- **Choosing experts by kk/ru routing works.** With 41% of experts removed, Kazakh perplexity grows by 10% and Russian by 27%. A random set of the same size breaks the model (×16 to ×1668 for Kazakh).
- **The calibration language matters.** Experts chosen on English text are much worse for kk/ru (×1.9 / ×3.3 at 38 experts). On this model, this supports [arXiv:2611.07654](https://arxiv.org/abs/2611.07654) rather than [arXiv:2610.15578](https://arxiv.org/abs/2610.15578).
- **English pays the price, as intended** (×2.1 at 38 experts). For English, frequency-based selection is no better than random: English uses experts broadly.
- **Router-weighted selection did not beat plain frequency.**
- **Russian degrades more than Kazakh.** Every layer keeps the same number of experts, but early layers need more of them (see notebook 01). Per-layer budgets come next.
- **Self-tests passed:** the mask matches physical removal exactly (choice agreement 1.0, weight difference 0).

⚠️ **Limitations:**
- Belebele multiple-choice accuracy of the *unpruned* model is near chance for kk/ru: 31% and 40% vs. 25% chance, with a strong bias toward answer "C". So accuracy cannot measure the damage here, and perplexity is the reliable signal.
- Perplexity is not the same as downstream task quality.
- This is one small model, and the mask does not yet save memory.

Raw numbers: [`experiments/02/`](experiments/02/).

## How it works

```
text in kk/ru ──► router hooks ──► which experts are picked, per layer
                                        │
                     keep the top-K experts per layer, mask the rest
                                        │
              evaluate: perplexity + reading comprehension (Belebele)
                vs. random experts and English-calibrated experts
```

Pruning is simulated with a router mask: dropped experts get a −∞ score.
The masked model's outputs are identical to a model with those experts physically deleted.
The notebook verifies this with built-in self-tests before measuring anything.

## Notebooks

| # | Notebook | What it does | Open |
|---|---|---|---|
| 01 | [`01_expert_profiling`](notebooks/01_expert_profiling.ipynb) | Which experts each language uses; how many could be removed | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/s0nakh/linguaprune/blob/main/notebooks/01_expert_profiling.ipynb) |
| 02 | [`02_prune_and_measure`](notebooks/02_prune_and_measure.ipynb) | Remove experts and measure quality on kk/ru/en against random and English baselines | [![Open in Kaggle](https://kaggle.com/static/images/open-in-kaggle.svg)](https://kaggle.com/kernels/welcome?src=https://github.com/s0nakh/linguaprune/blob/main/notebooks/02_prune_and_measure.ipynb) |

Notebooks are bilingual: English first, Russian translation below each text cell.

**Run on Kaggle:**
1. Click *Open in Kaggle*.
2. Set **Accelerator → GPU T4 x2** and **Internet → On**.
3. Run **Save Version → Save & Run All**.
4. Results land in `/kaggle/working/results`.

## Roadmap

- [x] 01 · Router profiling on OLMoE
- [x] 02 · Quality after pruning (Belebele kk/ru/en)
- [ ] 03 · Physically remove experts, save the smaller model, verify it matches the masked results, then 4-bit
- [ ] 04 · Same pipeline on Qwen3-30B-A3B
- [ ] REAP-style importance (gate × expert output norm) and per-layer budgets
- [ ] Prune + 4-bit vs. uniform 2–3-bit quantization **at equal memory**
- [ ] Larger MoE models for Kazakh/Russian
- [ ] `pip install linguaprune` CLI: profile → prune → export, for any language

**Our falsification test:** at equal memory, pruning + 4-bit must beat both uniform low-bit quantization and a smaller dense model on kk/ru quality. If it doesn't, we will publish that result.

## Development

Notebooks are edited as plain Python (`# %%` cells) and converted:

```bash
python3 scripts/py_to_ipynb.py notebooks/02_prune_and_measure.py
```

Notebooks are bilingual: every text cell is in English first, with a Russian translation below.

## Citation

```bibtex
@software{linguaprune2026,
  title  = {LinguaPrune: Language-aware expert pruning for MoE LLMs},
  author = {Akhmetova, Sophia},
  year   = {2026},
  url    = {https://github.com/s0nakh/linguaprune}
}
```

## License

[Apache-2.0](LICENSE). Models and datasets keep their own licenses. For example, KazMMLU is CC-BY-NC-4.0 and Belebele is CC-BY-SA-4.0.
