# Related work

*Survey as of 2026-11-10 · [Русская версия](prior-art.ru.md)*

Sources: arXiv abstracts and GitHub/Hugging Face pages, each checked by hand. Numbers are quoted from the abstracts; full papers have not been read yet.

## Closest work

| Work | What it shows | Link |
|---|---|---|
| English Calibration Erodes Lithuanian (2026-11) | REAP pruning of 25% of experts. With English calibration, Lithuanian loses 21.4 points while English loses 1.7. Lithuanian calibration recovers most of the loss. **Closest to this project.** | [arXiv:2611.07654](https://arxiv.org/abs/2611.07654) |
| Prune, Don't Merge (2026-11) | Pruning beats merging. Qwen3-30B keeps 93.6% of quality with 25% of experts removed. | [arXiv:2611.12240](https://arxiv.org/abs/2611.12240) |
| Towards Robust Calibration Dataset for Domain-General MoE Pruning (2026-10) | **The opposite finding:** English-only calibration transfers to 7 languages. | [arXiv:2610.15578](https://arxiv.org/abs/2610.15578) |
| Extracting Translation Specialists (2026) | Experts relevant to translation; 50% pruned with negligible loss. | [arXiv:2605.28042](https://arxiv.org/abs/2605.28042) |
| Memory-efficient NLLB-200 (2022) | Language-specific expert pruning in a machine-translation MoE, up to 80% of experts. | [arXiv:2212.09811](https://arxiv.org/abs/2212.09811) |
| REAP (Cerebras, 2025) | Expert score = gate value × expert output norm. Apache-2.0 code. | [arXiv:2510.13999](https://arxiv.org/abs/2510.13999), [GitHub](https://github.com/CerebrasResearch/reap) |
| How to Score Experts (2026) | Task-specific pruning benefits from keeping frequency and gate information. | [arXiv:2606.15716](https://arxiv.org/abs/2606.15716) |
| MESA (2026-09) | Frequency-based and activation-based criteria fail on different domains. Tested on OLMoE-1B-7B. | [arXiv:2609.04453](https://arxiv.org/abs/2609.04453) |
| MoEXBench (2026-08) | Pruning + quantization: pruning is the main source of degradation, and the two effects do not simply add up. | [arXiv:2608.21693](https://arxiv.org/abs/2608.21693) |
| Collaborative Compression (2025) | DeepSeek-V3: pruning + mixed precision beats uniform low-bit quantization at equal memory. | [arXiv:2509.25689](https://arxiv.org/abs/2509.25689) |

Routing analyses:
- Language-specific experts sit in early and late layers; middle layers are shared across languages ([arXiv:2510.04694](https://arxiv.org/abs/2510.04694), [arXiv:2601.14050](https://arxiv.org/abs/2601.14050)).
- Low-resource languages collapse onto a few experts in deep layers ([arXiv:2605.17598](https://arxiv.org/abs/2605.17598)). Our notebook 01 shows the same pattern for Kazakh.

## What is new here

**Already done by others:**
- keeping the experts that language X needs;
- calibrating REAP on the target language;
- combining pruning with quantization.

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
