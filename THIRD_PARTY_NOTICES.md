# Third-party notices

LinguaSqueeze's own code is licensed under [Apache-2.0](LICENSE). The research in this repository uses the software, models and datasets below. They keep their own licenses; nothing from them is redistributed here except the measured scores, which are our own results.

Licenses were checked against each project's repository or Hugging Face card on 2026-11-15.

## Software

| Component | Used for | License |
|---|---|---|
| [PyTorch](https://github.com/pytorch/pytorch) | Model loading, GPTQ, evaluation | BSD-3-Clause style |
| [NumPy](https://github.com/numpy/numpy) | Numerics | BSD-3-Clause |
| [pandas](https://github.com/pandas-dev/pandas) | Score tables | BSD-3-Clause |
| [Transformers](https://github.com/huggingface/transformers) | Models and tokenizers | Apache-2.0 |
| [Datasets](https://github.com/huggingface/datasets) | Evaluation and calibration data | Apache-2.0 |
| [huggingface_hub](https://github.com/huggingface/huggingface_hub) | Downloads | Apache-2.0 |
| [safetensors](https://github.com/huggingface/safetensors) | Weight files | Apache-2.0 |
| [GPTQModel](https://github.com/ModelCloud/GPTQModel) | Running the compressed checkpoints (notebook 08) | Apache-2.0 |
| [llama.cpp](https://github.com/ggml-org/llama.cpp) | GGUF tools and inference (notebooks 09–10) | MIT |
| [llama-cpp-python](https://github.com/abetlen/llama-cpp-python) | Python bindings for llama.cpp | MIT |
| [Matplotlib](https://matplotlib.org) | Figures | Matplotlib License (PSF-based) |
| [pytest](https://github.com/pytest-dev/pytest) | Tests | MIT |

## Models

| Model | Used for | License |
|---|---|---|
| [Qwen/Qwen3-30B-A3B-Instruct-2507](https://huggingface.co/Qwen/Qwen3-30B-A3B-Instruct-2507) | Main compressed model | Apache-2.0 |
| [allenai/OLMoE-1B-7B-0125](https://huggingface.co/allenai/OLMoE-1B-7B-0125) | Early experiments (notebooks 01–05) | Apache-2.0 |
| [unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF](https://huggingface.co/unsloth/Qwen3-30B-A3B-Instruct-2507-GGUF) | Comparison files and the Q8_0 source of notebook 10 | Apache-2.0 |
| [bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF](https://huggingface.co/bartowski/Qwen_Qwen3-30B-A3B-Instruct-2507-GGUF) | Comparison file (notebook 09) | No license declared on the card; a quantization of Qwen3 (Apache-2.0). Evaluated only, not redistributed |

Qwen, OLMoE, unsloth and bartowski are names of their respective owners. They are used here only to identify the models we measured.

## Datasets

| Dataset | Used for | License | Notes |
|---|---|---|---|
| [facebook/belebele](https://huggingface.co/datasets/facebook/belebele) | Perplexity and reading comprehension in kk, ru, en | CC-BY-SA-4.0 | Scores only; no passages are redistributed |
| [MBZUAI/KazMMLU](https://huggingface.co/datasets/MBZUAI/KazMMLU) | Knowledge exam in Kazakh and Russian | **CC-BY-NC-4.0** | Non-commercial research use only |
| [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) | Calibration text (kk, ru, en) | CC-BY-SA-3.0 and GFDL | Calibration only; never part of the exam |

### Dataset citations

- Bandarkar et al. *The Belebele Benchmark: a Parallel Reading Comprehension Dataset in 122 Language Variants.* ACL 2024.
- Togmanov et al. *KazMMLU: Evaluating Language Models on Kazakh, Russian, and Regional Knowledge of Kazakhstan.* ACL 2025.
