# Research log

*[Русская версия](research-log.ru.md) · [Back to README](../README.md)*

A dated record of every experiment: the question, the setup, the result and the decision it led to.
Negative results stay in. Raw outputs are in [`experiments/`](../experiments/), and each entry links to its notebook and data.

**Fixed conventions**
- Exam: [Belebele](https://huggingface.co/datasets/facebook/belebele), the same 150 passages in kk / ru / en. It is never used for calibration.
- Calibration: Wikipedia (2023-11-01 snapshot).
- Primary metric: passage perplexity relative to the unpruned / unquantized model.
- Every pruning or quantization result is compared against **controls at equal memory**: random experts and English-chosen experts.
- Memory figures for quantization are **analytic** (bits per weight plus group scales). Quantization is simulated unless stated otherwise.

---

## 2026-11-10 · 01 · Which experts does each language use?

- **Question.** Does an MoE router use a narrower set of experts for Kazakh than for English?
- **Setup.**
  - Model: OLMoE-1B-7B-0125 (16 layers × 64 experts, top-8).
  - Data: 100k Wikipedia tokens each for kk, ru, en and zh.
  - We count router picks and find the smallest expert set that covers 90% of picks.
  - Estimates use held-out articles and a bootstrap.
- **Result.**
  - kk needs **22.5** of 64 experts, ru 35.9, zh 39.5, en 47.8.
  - kk ∪ ru needs 37.7, with a 95% CI of 37.3–38.4.
  - Held-out coverage: 96% (kk) and 91% (ru).
  - Deep layers concentrate further: layer 14 needs 12 experts for kk.
  - A second independent run reproduced every number and every per-layer expert set exactly.
- **Decision.** Routing is language-specific enough to try pruning. Selection frequency is not the same as importance, so quality must be measured next.
- [Notebook](../notebooks/01_expert_profiling.ipynb) · [data](../experiments/01/)

## 2026-11-11 · 02 · Does pruning by kk+ru routing preserve quality?

- **Question.** Removing rarely used experts (simulated with a router mask), how much do kk/ru get worse, compared with random and English-chosen experts?
- **Setup.**
  - Keep 48 / 38 / 32 / 24 of 64 experts per layer.
  - Selection methods:
    - kk+ru frequency;
    - kk+ru frequency weighted by router score;
    - English frequency;
    - 5 random draws.
  - A self-test checks that the mask is equivalent to physical removal.
- **Result** (perplexity relative to the original, 38/64 experts kept):

  | Language | kk+ru selection | random | English-chosen |
  |---|---|---|---|
  | kk | ×1.10 | ×16–1668 | ×1.86 |
  | ru | ×1.27 | — | ×3.26 |

  - Router-weighted selection gave no gain over plain frequency.
  - Belebele accuracy is near chance for the base model in kk/ru (31% / 40%), so accuracy cannot measure the damage.
- **Decision.** Language-aware selection clearly works relative to the controls. The absolute loss for Russian is large, so physical removal comes next.
- [Notebook](../notebooks/02_prune_and_measure.ipynb) · [data](../experiments/02/)

## 2026-11-11 · 03 · Physical removal

- **Question.** Does a physically smaller checkpoint behave exactly like the mask?
- **Setup.**
  - Delete the 26 dropped experts per layer from the weights.
  - Save with `save_pretrained` and reload with `from_pretrained`.
  - Compare against the mask.
- **Result.**
  - Size: 6.92B → 4.30B parameters (−38%), 13.8 → 8.6 GB in fp16.
  - Perplexity matches the mask within 0.07%. Positions where the model is confident agree 100%, and the reloaded model is bit-identical.
  - Greedy samples (anecdotal): the Russian capital answer flips from "Astana" to "Almaty".
- **Decision.** Deleting experts deletes knowledge. Instead, keep every expert and compress the rarely used ones harder.
- [Notebook](../notebooks/03_remove_and_save.ipynb) · [data](../experiments/03/)

## 2026-11-11 · 04 · Language-aware mixed precision (round-to-nearest)

- **Question.** With every expert kept, can kk+ru-guided bit allocation (4/3/2 bits) reach the target?
- **Target**, fixed before the run: ≥3.5× smaller than fp16, with kk and ru perplexity growing ≤ 2%.
- **Result.** The target was **not met**.
  - Uniform 4-bit RTN already costs +6.4% on kk.
  - At equal memory, kk+ru allocation beats English allocation: kk +8.8% vs +16.5% at 4/3 bits. It beats English and random by ~15–20× at 4/3/2 bits.
- **Decision.** Allocation works; the quantizer is the bottleneck. Switch to GPTQ calibrated on kk+ru text.
- [Notebook](../notebooks/04_language_aware_quantization.ipynb) · [data](../experiments/04/)

## 2026-11-12 · Engineering: layer-by-layer streaming

- **Problem.** Qwen3-30B-A3B weighs ~61 GB in 16-bit, while free Kaggle has 2 × 15 GB GPUs.
- **Solution.**
  - `linguaprune.streaming` loads one decoder layer at a time from the safetensors files.
  - It carries hidden states for several configurations side by side and scores sequences without materializing full-vocabulary logits.
- **Verification.** Continuous-integration tests compare streamed scores with the normal forward pass on tiny real OLMoE and Qwen3-MoE models. They also check that each configuration's calibration never sees the exam.
- **Incident.** On Kaggle's T4 (CUDA 13), batched `cholesky_inverse` (MAGMA) crashed with an illegal memory access. Now the inverse factor is computed per matrix via triangular solves, while the GPTQ column loop stays batched. This was caught by a 2-layer dry run before the full run.

## 2026-11-12 · 05 · GPTQ calibrated on kk+ru, streamed

- **Question.** Does a data-aware quantizer close the gap?
- **Setup.**
  - GPTQ per expert: a Hessian built only from the calibration tokens routed to that expert, and the experts of a layer batched together.
  - Attention in 8-bit RTN.
  - Calibration: 64 kk + 64 ru Wikipedia articles, or English for the control.
- **Result** (OLMoE, perplexity growth):

  | Configuration | kk | ru | Size vs fp16 |
  |---|---|---|---|
  | RTN 4-bit | +6.6% | +4.5% | 3.4× |
  | **GPTQ 4-bit, kk+ru calibration** | **+1.5%** | **+1.5%** | 3.4× |
  | GPTQ kk+ru 4/3 | +2.2% | +4.4% | 3.9× |
  | GPTQ English 4/3 (equal memory) | +20.2% | +13.4% | 3.9× |
  | GPTQ kk+ru 4/3/2 | +2.0% | +18.4% | 4.6× |

- **Decision.** GPTQ cuts the loss 3–4× and the calibration language matters strongly. Kazakh stays near 2% even at 4.6×; Russian is the weak spot, because Kazakh concentrates on few experts and takes the 4-bit slots. The next run adds a variant that gives 4 bits to the experts covering 90% of kk+ru traffic.
- [Notebook](../notebooks/05_gptq_streaming.ipynb) · [data](../experiments/05/)

## 2026-11-12 · 06 · Qwen3-30B-A3B: does the model know Kazakh?

- **Question.** Before compressing, is there knowledge worth protecting?
- **Setup.**
  - Qwen3-30B-A3B-Instruct-2507: 48 layers × 128 experts, 30.5B parameters.
  - Streamed from the official Kaggle model files on one T4; no compression.
- **Result.**
  - The whole model was processed in **15 minutes**.
  - KazMMLU: **66.8%** kk and **71.4%** ru, on ~380 subject-balanced questions each (chance ≈ 20–25%).
  - Belebele accuracy: 92.7% kk.
  - Factual probes: 3/3 in every language.
- **Decision.** Yes. The compression run starts with:
  - original;
  - GPTQ 4-bit;
  - kk+ru 4/3 at 80% and at 90% traffic;
  - English 4/3 at equal memory;
  - kk+ru 4/3/2.
- [Notebook](../notebooks/06_qwen3_language_aware_gptq.ipynb) · [baseline data](../experiments/06/baseline/)
