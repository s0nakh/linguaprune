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
  - Belebele accuracy: 77.3% kk, 92.7% ru and en. *(Corrected 2026-11-12: an earlier version of this entry gave 92.7% for kk.)*
  - Factual probes: 3/3 in every language.
- **Decision.** Yes. The compression run starts with:
  - original;
  - GPTQ 4-bit;
  - kk+ru 4/3 at 80% and at 90% traffic;
  - English 4/3 at equal memory;
  - kk+ru 4/3/2.
- [Notebook](../notebooks/06_qwen3_language_aware_gptq.ipynb) · [baseline data](../experiments/06/baseline/)

## 2026-11-12 · 06 · Qwen3-30B-A3B compressed 4×

- **Question.** Does language-aware GPTQ reach the target on a model that is strong in Kazakh?
- **Target**, fixed before the run: ≥3.5× smaller than 16-bit; kk and ru perplexity growth ≤ 2%; KazMMLU not lower beyond noise.
- **Setup.**
  - Qwen3-30B-A3B-Instruct-2507, streamed layer by layer on one Kaggle T4 (145 min for 6 configurations).
  - Experts: GPTQ, group 128, calibrated on 64 kk + 64 ru Wikipedia articles (English for the control). Experts that receive fewer than 16 calibration tokens fall back to round-to-nearest.
  - Bits per expert from live router traffic on the calibration text: experts covering 80% or 90% of kk+ru traffic get 4 bits, the rest 3 bits.
  - Exam: Belebele perplexity and accuracy (150 per language), KazMMLU (370 kk + 392 ru), factual probes. Paired tests against the original, question by question.
- **Result.**

  | Configuration | Memory | kk ppl | ru ppl | KazMMLU kk | KazMMLU ru | Belebele kk / ru |
  |---|---|---|---|---|---|---|
  | original | 61.1 GB | — | — | 66.8% | 71.4% | 77.3 / 92.7% |
  | GPTQ 4-bit | 17.6 GB (3.5×) | +1.7% | +1.9% | −1.6 pt | −0.5 pt | 0.0 / +0.7 pt |
  | **kk+ru 4/3 @90%** | **15.5 GB (3.9×)** | **+0.9%** | **+1.0%** | −2.7 pt | +0.3 pt | 0.0 / 0.0 pt |
  | **kk+ru 4/3 @80%** | **15.0 GB (4.1×)** | **−0.4%** | **−0.9%** | −1.9 pt | −2.0 pt | 0.0 / −3.3 pt |
  | English 4/3 @90% (control) | 15.5 GB (3.9×) | +14.3% | +8.3% | +0.3 pt | +0.8 pt | −2.7 / −1.3 pt |
  | kk+ru 4/3/2 | 13.2 GB (4.6×) | +1.9% | +7.3% | −40.8 pt | −47.7 pt | −41.3 / −54.0 pt |

  - **Paired tests** (bootstrap 95% CI, McNemar):
    - kk+ru 4/3 @90%, KazMMLU kk: −2.7 pt [−5.4, 0.0], 18 questions lost and 8 gained, p = 0.08.
    - kk+ru 4/3 @80%, KazMMLU kk: −1.9 pt [−5.1, +1.1], p = 0.32; KazMMLU ru: −2.0 pt [−4.6, +0.3], p = 0.17; Belebele ru: −3.3 pt [−7.3, +0.7], p = 0.18.
    - No difference for the two target configurations is significant at 0.05. Every 4/3/2 difference is (p < 0.001).
  - Factual probes: 3/3 in every language for every configuration.
  - **Target met** by kk+ru 4/3 at 80% and at 90%.
- **Observations.**
  - At equal memory, English calibration costs Kazakh perplexity 16× more than kk+ru calibration (+14.3% vs. +0.9%).
  - But KazMMLU does not separate the two: the English control keeps KazMMLU (+0.3 pt), while every kk+ru configuration trends 2–3 points lower on kk. With 370 questions we cannot tell whether this is noise.
  - 2-bit experts break the model even though Kazakh perplexity stays at +1.9%. Perplexity on one language is not a safe quality check on its own.
  - Negative perplexity growth at 80% likely reflects calibration on similar (Wikipedia-style) text; it is not a claim that the compressed model is better.
- **Decision.**
  - Recommended configuration: kk+ru 4/3 at 90% (Belebele unchanged in all languages).
  - Next: a real low-bit file (measured size and speed, not analytic); then the full KazMMLU to resolve the kk trend.
- [Notebook](../notebooks/06_qwen3_language_aware_gptq.ipynb) · [data](../experiments/06/full/) · [figure script](../scripts/plot_qwen3_result.py)

## 2026-11-12 · 07 · A real compressed file

- **Question.** Do the 06 results survive when the model is written as real low-bit files and read back from disk?
- **Setup.**
  - Same model, calibration and exam as 06; GPTQ re-run with linguaprune 0.5.0, whose quantization grid stores fp16 scales and integer zero points exactly as the file does.
  - Written in the standard GPTQ v1 layout: experts 4 or 3 bits (group 128), attention 8 bits, embeddings and norms 16 bits. Per-module bit widths are declared in the `dynamic` field of `quantization_config`.
  - Every quantized row of the table is scored **from the written file**, not from the in-memory model. One Kaggle T4, 116 min including upload.
- **Result.**

  | Configuration | On disk | kk ppl | ru ppl | KazMMLU kk | KazMMLU ru | Belebele kk / ru / en |
  |---|---|---|---|---|---|---|
  | original | 61.1 GB | — | — | 66.8% | 71.4% | 77.3 / 92.7 / 92.7% |
  | **kk+ru 4/3 @90%** | **15.24 GB (4.0×)** | **+1.3%** | **+0.4%** | 65.1% | 70.9% | 78.7 / 91.3 / 93.3% |
  | kk+ru 4-bit | 17.39 GB (3.5×) | +1.2% | +1.4% | 64.9% | 71.7% | 76.0 / 92.7 / 94.0% |

  - Largest score difference between the in-memory model and the file: **0.0** for both configurations.
  - Factual probes: 3/3 in every language.
  - **Target met from a real file**: 4.0× smaller, kk and ru perplexity within 2%.
- **Observations.**
  - The numbers move slightly from 06 (kk +0.9% → +1.3%, ru +1.0% → +0.4%) because the quantization grid changed to match the file format; that is run-to-run variation of GPTQ, not file loss.
  - The KazMMLU kk gap is −1.6 pt here vs. −2.7 pt in 06; the full KazMMLU is still needed to settle it.
  - Not tested yet: loading in vLLM and other runtimes, and speed.
- **Decision.** Publish both checkpoints on Hugging Face; next, test runtime compatibility and compare against popular 4-bit versions on the same exam.
- [Notebook](../notebooks/07_real_gptq_checkpoint.ipynb) · [data](../experiments/07/full/)

## 2026-11-12 · 08 · The compressed model in a standard runtime

- **Question.** Does a runtime we do not control read the files correctly and answer questions?
- **Setup.**
  - [GPTQModel](https://github.com/ModelCloud/GPTQModel) 7.6.0 on transformers 5.20, torch kernel, two Kaggle T4.
  - GPTQModel refuses asymmetric checkpoints in the classic GPTQ layout from other tools, so each download is converted to its `gptq_v2` layout first (zero points stored without the −1 offset; nothing else changes). LinguaPrune 0.5.1 adds this conversion.
  - Tiny check: a random 2-layer Qwen3-MoE with mixed 3/4/8-bit modules, written by LinguaPrune and loaded by GPTQModel.
  - Real models: 40 kk + 40 ru Belebele passages re-scored and compared with notebook 07 on the same passages; five questions in kk, ru and en, greedy decoding.
- **Result.**
  - Tiny check: 0.00003 nats per token between LinguaPrune and GPTQModel.

  | Model | GPU memory | kk ppl vs. notebook 07 | ru ppl vs. notebook 07 | Speed |
  |---|---|---|---|---|
  | kk+ru 4-bit | 17.4 GB | +0.01% | −0.08% | 1.6 tokens/s |
  | kk+ru 4/3-bit | 15.3 GB | −0.04% | +0.01% | 1.6 tokens/s |

  - Both models answer correctly and fluently in Kazakh, Russian and English (capital, translation, photosynthesis, rivers, Abai). Answers are in [`experiments/08/`](../experiments/08/).
  - One factual slip: the 4/3-bit model gives Abai's death year as 1908 (correct: 1904; the 4-bit model gets it right).
- **Observations.**
  - The mixed 4/3-bit file works in a standard runtime: GPTQModel reads every module at its declared width.
  - Speed is low because the portable torch kernel dequantizes every expert on every call; faster kernels and vLLM are not tested yet.
- **Decision.** Re-upload both checkpoints in the `gptq_v2` layout with usage instructions, then try vLLM and a GGUF build for Ollama / LM Studio users.
- [Notebook](../notebooks/08_run_with_gptqmodel.ipynb) · [data](../experiments/08/)
