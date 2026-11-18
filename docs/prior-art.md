# Related work

*Survey as of 2026-11-19 · [Русская версия](prior-art.ru.md) · [Қазақша нұсқасы](prior-art.kk.md)*

Sources: arXiv and GitHub/Hugging Face pages, each checked by hand. The full texts of the works cited in the preprint were read (audits of 2026-11-17 and 2026-11-19); the other rows rest on abstracts.

## Closest work

| Work | What it shows | Link |
|---|---|---|
| English Calibration Erodes Lithuanian (2026-11) | REAP pruning of 25% of experts. With English calibration, Lithuanian loses 21.4 points while English loses 1.7. Lithuanian calibration recovers most of the loss. **Closest to this project.** | [arXiv:2611.07654](https://arxiv.org/abs/2611.07654) |
| Prune, Don't Merge (2026-11) | Pruning beats merging on five multilingual production tasks. At 25% of experts removed, REAP keeps 93.6% of Qwen3-30B's generative quality. | [arXiv:2611.12240](https://arxiv.org/abs/2611.12240) |
| Towards Robust Calibration Dataset for Domain-General MoE Pruning (2026-10) | **The opposite finding:** English-only calibration transfers to 7 languages. | [arXiv:2610.15578](https://arxiv.org/abs/2610.15578) |
| Extracting Translation Specialists (2026) | Experts relevant to translation; 50% pruned with negligible loss. | [arXiv:2605.28042](https://arxiv.org/abs/2605.28042) |
| Memory-efficient NLLB-200 (2022) | Language-specific expert pruning in a machine-translation MoE, up to 80% of experts. | [arXiv:2212.09811](https://arxiv.org/abs/2212.09811) |
| REAP (Cerebras, 2025) | Expert score = gate value × expert output norm. Apache-2.0 code. | [arXiv:2510.13999](https://arxiv.org/abs/2510.13999), [GitHub](https://github.com/CerebrasResearch/reap) |
| How to Score Experts (2026) | Task-specific pruning benefits from keeping frequency and gate information. | [arXiv:2606.15716](https://arxiv.org/abs/2606.15716) |
| MESA (2026-09) | With over-dispersed routing, perplexity does not predict downstream accuracy and no single scoring criterion wins on all domains (mainly gpt-oss-20B; also OLMoE-1B-7B). | [arXiv:2609.04453](https://arxiv.org/abs/2609.04453) |
| MoEXBench (2026-08) | Pruning + quantization: pruning is the main source of degradation, and the two effects do not simply add up. | [arXiv:2608.21693](https://arxiv.org/abs/2608.21693) |
| Collaborative Compression (2025) | DeepSeek-V3: pruning + mixed precision beats uniform low-bit quantization at equal memory. | [arXiv:2509.25689](https://arxiv.org/abs/2509.25689) |
| Calibrating Beyond English (2026-01) | GPTQ and AWQ on 10 languages (Llama3.1 8B, Qwen2.5 7B): non-English and multilingual calibration lowers perplexity against English-only; matching the evaluation language helps most. Dense models. | [arXiv:2601.18306](https://arxiv.org/abs/2601.18306) |
| The Uneven Impact of Post-Training Quantization in Machine Translation (2025) | 55 languages, five LLMs from 1.7B to 70B: 4-bit mostly keeps high-resource languages, while low-resource and typologically diverse languages degrade, especially at 2 bits. | [arXiv:2508.20893](https://arxiv.org/abs/2508.20893) |
| QuantMoE-Bench (2024) | Fine-grained mixed precision for MoE quantization; different MoE structures need different numbers of bits. | [arXiv:2406.08155](https://arxiv.org/abs/2406.08155) |
| MC, Mixture Compressor (2024) | Bit-width allocation across experts as a linear program over expert importance, plus dynamic pruning. | [arXiv:2410.06270](https://arxiv.org/abs/2410.06270) |
| MoPEQ (2025) | Per-expert bit width from Hessian-trace sensitivity instead of activation frequency. | [arXiv:2509.02512](https://arxiv.org/abs/2509.02512) |
| GEMQ (2026-05) | Global expert-level mixed-precision allocation. | [arXiv:2605.23078](https://arxiv.org/abs/2605.23078) |
| Q-Strata (2026-08) | Hierarchical bit allocation for MoE quantization. | [arXiv:2608.30564](https://arxiv.org/abs/2608.30564) |
| AlphaQ (2026-06) | Calibration-free bit allocation from weight-spectrum heavy-tailedness; reports gains over calibration-based baselines at matched bit budgets. | [arXiv:2606.04980](https://arxiv.org/abs/2606.04980) |
| Limitations of language-targeted pruning (Kurz et al., TACL, 2025) | Dense models, Wanda and SparseGPT pruning with 6 calibration languages, plus a GPTQ experiment (4 and 8 bits, LLaMA-3-8B): target-language calibration keeps perplexity but does not consistently improve downstream tasks. No Kazakh, no MoE. | [arXiv:2408.14398](https://arxiv.org/abs/2408.14398) |
| Deep Language Geometry (Shamrai and Hamolia, 2025) | SparseGPT importance scores over 106 languages (Kazakh and Russian included) give a metric space of languages; no model is compressed. | [arXiv:2508.11676](https://arxiv.org/abs/2508.11676) |
| BitsMoE (2026-09) | Integer-program bit allocation over spectral components of experts; English C4 calibration; no low-resource language. | [arXiv:2606.00079](https://arxiv.org/abs/2606.00079) |
| DuoRate (2026-10) | Two-rate per-expert precision allocation for serving; in its decode-rate formulation routing frequency cancels from the per-expert choice; English benchmarks. | [arXiv:2610.13328](https://arxiv.org/abs/2610.13328) |
| Routing is not a quantization proxy (Liu, 2026-10) | Qwen-3.6-35B-A3B on code tasks: layer-wise allocation guided by routing statistics does not beat uniform or random allocation at equal bits. Per-expert allocation not tested. | [arXiv:2610.20826](https://arxiv.org/abs/2610.20826) |
| Language Routing Isolation (Zheng et al., 2026) | Phi-3.5-MoE and Qwen3-30B-A3B: high- and low-resource languages activate largely disjoint experts; subnetwork fine-tuning, no compression. | [arXiv:2604.03592](https://arxiv.org/abs/2604.03592) |

Routing analyses:
- Language-specific experts sit in early and late layers; middle layers are shared across languages ([arXiv:2510.04694](https://arxiv.org/abs/2510.04694), [arXiv:2601.14050](https://arxiv.org/abs/2601.14050)).
- Low-resource languages collapse onto a few experts in deep layers ([arXiv:2605.17598](https://arxiv.org/abs/2605.17598)). Our notebook 01 shows the same pattern for Kazakh.

## What is new here

**Already done by others:**
- keeping the experts that language X needs;
- calibrating REAP on the target language;
- combining pruning with quantization;
- giving different experts different bit widths (QuantMoE-Bench, MC, MoPEQ, GEMQ, Q-Strata, AlphaQ);
- non-English calibration for quantizing dense models (Calibrating Beyond English).

**Open questions this project addresses:**
1. Kazakh and Russian (a Turkic language and Cyrillic script): no published work yet.
2. One expert set for kk and ru together: does high-resource Russian protect low-resource Kazakh?
3. At equal memory: prune + 4-bit vs. uniform 2–3-bit quantization, measured on a specific language's quality.
4. Large MoE models for Kazakh/Russian, evaluated on KazMMLU.

## What notebook 02 does

- Expert removal is simulated with a router mask. For the forward pass this is mathematically equivalent to deletion, and built-in self-tests check it.
- Selection methods are compared at an equal number of kept experts:
  1. kk+ru frequency;
  2. router-weighted kk+ru frequency;
  3. English frequency (control);
  4. five random draws.
- Quality is measured with perplexity and accuracy on Belebele (`kaz_Cyrl`, `rus_Cyrl`, `eng_Latn`).
- Next: a REAP-style criterion (gate × expert output norm) and per-layer budgets.

## Evaluation dataset licenses

| Dataset | License |
|---|---|
| [KazMMLU](https://huggingface.co/datasets/MBZUAI/KazMMLU) (MBZUAI) | CC-BY-NC-4.0 |
| [Belebele](https://huggingface.co/datasets/facebook/belebele) (Meta) | CC-BY-SA-4.0 |
| [FLORES+](https://huggingface.co/datasets/openlanguagedata/flores_plus) | CC-BY-SA-4.0 (gated) |
| [MERA](https://huggingface.co/datasets/MERA-evaluation/MERA) (Russian) | MIT |
