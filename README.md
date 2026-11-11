<div align="center">

<img src="assets/banner.png" alt="LinguaPrune" width="720">

**Language-aware expert pruning for Mixture-of-Experts LLMs.**
Keep the experts your languages actually use, drop the rest, then quantize.
First target: Kazakh 🇰🇿 + Russian.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.11%2B-3776AB?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.x-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Transformers](https://img.shields.io/badge/%F0%9F%A4%97%20Transformers-5.x-FFD21E)](https://github.com/huggingface/transformers)
[![Runs on Kaggle](https://img.shields.io/badge/runs%20on-Kaggle%20T4%20x2-20BEFF?logo=kaggle&logoColor=white)](https://www.kaggle.com/)
[![Status](https://img.shields.io/badge/status-research%20preview-orange)](#roadmap)

[Русская версия](README.ru.md) · [Related work](docs/prior-art.md)

</div>

---

## Why

Sovereign and multilingual MoE models are huge. A 247B-parameter MoE, for example, needs roughly 494 GB in BF16, or about 7 server GPUs.
But a model serving Kazakh and Russian users rarely calls many of its experts.
LinguaPrune measures **which experts a language really needs** and removes the rest, so the model fits on fewer GPUs.

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

Raw numbers: [`experiments/01/summary.json`](experiments/01/summary.json).

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

**Run on Kaggle:**
1. Click *Open in Kaggle*.
2. Set **Accelerator → GPU T4 x2** and **Internet → On**.
3. Run **Save Version → Save & Run All**.
4. Results land in `/kaggle/working/results`.

Notebook comments are currently in Russian; an English translation is planned.

## Roadmap

- [x] 01 · Router profiling on OLMoE
- [ ] 02 · Quality after pruning (Belebele kk/ru/en)
- [ ] 03 · Same pipeline on Qwen3-30B-A3B
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
