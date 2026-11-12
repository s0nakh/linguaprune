# %% [markdown]
# ![LinguaPrune](https://raw.githubusercontent.com/s0nakh/linguaprune/main/assets/banner.png)
#
# # 08 · Can people run it? The compressed Qwen3-30B-A3B in a standard runtime
#
# **Author:** Sophia Akhmetova ([@s0nakh](https://github.com/s0nakh)) · Apache-2.0
#
# Notebook 07 wrote the compressed models as standard GPTQ files and scored them with LinguaPrune's own reader.
# Here a **standard runtime that LinguaPrune does not control** loads the same files:
# [GPTQModel](https://github.com/ModelCloud/GPTQModel) on top of transformers.
#
# GPTQModel refuses asymmetric checkpoints in the classic GPTQ layout from other tools, so each download is first
# converted to its `gptq_v2` layout (`linguaprune.export.convert_to_gptq_v2`: zero points stored without the −1
# offset; nothing else changes).
#
# 1. **Tiny check (no download, ~3 min).** A small random Qwen3-MoE is written by LinguaPrune with mixed 3/4/8-bit
#    modules and loaded by GPTQModel; both must give the same scores. This catches a runtime that loads the file
#    but silently reads it wrong.
# 2. **The real models.** For each checkpoint on Hugging Face:
#    - load it with GPTQModel on two T4 GPUs;
#    - **score 40 Kazakh + 40 Russian Belebele passages** and compare with the numbers notebook 07 measured from the
#      same file (agreement within 2% means the runtime reads the file correctly);
#    - **ask questions** in Kazakh, Russian and English and measure the speed.
#
# **Run:** Add-ons → Secrets → `HF_TOKEN` (the models are private for now); Accelerator **GPU T4 x2**; Internet On.
# Without the secret only the tiny check runs.
#
# ---
#
# 🇷🇺 **08 · Можно ли этим пользоваться? Сжатая Qwen3-30B-A3B в обычной программе**
#
# Ноутбук 07 записал сжатые модели в стандартном формате GPTQ и оценил их собственным читателем LinguaPrune.
# Здесь те же файлы загружает **обычная программа, которую LinguaPrune не контролирует**: GPTQModel поверх transformers.
#
# GPTQModel не принимает несимметричные файлы в классическом формате GPTQ от других программ, поэтому каждая скачанная
# модель сначала переводится в её вариант `gptq_v2` (`convert_to_gptq_v2`: нулевые точки хранятся без сдвига на −1;
# больше ничего не меняется).
#
# 1. **Маленькая проверка (без скачивания, ~3 мин).** LinguaPrune записывает маленькую случайную Qwen3-MoE со смесью
#    3/4/8 бит, GPTQModel её загружает; оценки должны совпасть. Так ловится программа, которая грузит файл без ошибки,
#    но молча читает его неправильно.
# 2. **Настоящие модели.** Для каждой модели на Hugging Face:
#    - загружаем через GPTQModel на двух T4;
#    - **оцениваем 40 казахских и 40 русских отрывков Belebele** и сравниваем с цифрами ноутбука 07 для того же файла
#      (совпадение в пределах 2% значит, что программа читает файл правильно);
#    - **задаём вопросы** на казахском, русском и английском и меряем скорость.
#
# **Запуск:** Add-ons → Secrets → `HF_TOKEN` (модели пока приватные); Accelerator **GPU T4 x2**; Internet On.
# Без секрета выполняется только маленькая проверка.

# %%
import subprocess
import sys

LINGUAPRUNE_REF = "d54eecfc6c8215a9ef69f752e2b79d521c572e3f"  # pinned commit of the linguaprune package (v0.5.1)
GPTQMODEL_VERSION = "7.6.0"  # the newest release that accepts the torch preinstalled on Kaggle
pip = [sys.executable, "-m", "pip", "install", "-q"]
subprocess.check_call(pip + [f"git+https://github.com/s0nakh/linguaprune@{LINGUAPRUNE_REF}"])
# --no-deps keeps Kaggle's numpy and torch; the small runtime dependencies are added explicitly
subprocess.check_call(pip + ["--no-deps", f"gptqmodel=={GPTQMODEL_VERSION}"])
subprocess.check_call(pip + ["device-smi>=0.5.7", "logbar>=0.4.14", "tokenicer>=0.0.14", "pypcre>=0.6.2",
                             "defuser>=0.0.28", "torchao>=0.18.0", "threadpoolctl>=3.6.0", "dill>=0.3.8"])

# %%
import gc
import json
import platform
import shutil
import time
import traceback
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import transformers
from transformers import AutoTokenizer

import gptqmodel
import linguaprune
from gptqmodel import BACKEND, GPTQModel
from linguaprune.belebele import build_example, load_eval_rows
from linguaprune.export import convert_to_gptq_v2

HF_USER = "s0nakh"
MODELS = {  # notebook 07 configuration name → Hugging Face repository
    "kk+ru 4-bit": "Qwen3-30B-A3B-Instruct-2507-kk-ru-GPTQ-4bit",
    "kk+ru 4/3-bit @90%": "Qwen3-30B-A3B-Instruct-2507-kk-ru-GPTQ-4-3bit",
}
SCORES_07 = "https://raw.githubusercontent.com/s0nakh/linguaprune/main/experiments/07/full/scores.csv"
EVAL_LANGUAGES = {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}  # same sample as notebook 07
N_EVAL, SEED = 150, 0
N_CHECK = 40  # Belebele passages per language re-scored by the runtime
MAX_PPL_MISMATCH = 0.02  # runtime vs. notebook 07 perplexity on the same passages
MAX_NEW_TOKENS = 120
QUESTIONS = [
    ("kk", "Қазақстанның астанасы қай қала? Бір сөйлеммен жауап бер."),
    ("kk", "Абай Құнанбайұлы кім болған? Қысқаша айтып бер."),
    ("ru", "Объясни в двух предложениях, что такое фотосинтез."),
    ("ru", "Переведи на казахский: «Сегодня хорошая погода»."),
    ("en", "Name three rivers in Kazakhstan."),
]

KAGGLE_WORKING = Path("/kaggle/working")
OUTPUT_DIR = (KAGGLE_WORKING if KAGGLE_WORKING.exists() else Path(".")) / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
DOWNLOAD_ROOT = Path("/kaggle/tmp") if KAGGLE_WORKING.exists() else Path("downloads")
DOWNLOAD_ROOT.mkdir(parents=True, exist_ok=True)
if not torch.cuda.is_available():
    raise RuntimeError("No GPU found. On Kaggle, enable Settings → Accelerator → GPU T4 x2.")
VERSIONS = {"python": platform.python_version(), "torch": torch.__version__,
            "transformers": transformers.__version__, "gptqmodel": gptqmodel.__version__,
            "linguaprune": linguaprune.__version__}
print(VERSIONS, f"| GPU x{torch.cuda.device_count()}")


def hf_token() -> str | None:
    """The HF_TOKEN Kaggle secret, or None. The value is never printed."""
    try:
        from kaggle_secrets import UserSecretsClient
        return UserSecretsClient().get_secret("HF_TOKEN")
    except Exception as error:  # secret not attached, or not running on Kaggle
        print(f"HF_TOKEN secret not available ({type(error).__name__}); only the tiny check will run.")
        return None


TOKEN = hf_token()
summary = {"versions": VERSIONS, "tiny_check": None, "models": {}}


def save_summary() -> None:
    (OUTPUT_DIR / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2))


def span_logprobs(model: torch.nn.Module, sequences: list[tuple[list[int], tuple[int, int]]]) -> list[float]:
    """Sum of log-probabilities of each sequence's span, one sequence at a time (no padding effects)."""
    device = next(model.parameters()).device
    out = []
    with torch.inference_mode():
        for ids, (start, end) in sequences:
            logits = model(input_ids=torch.tensor([ids], device=device)).logits[0, start - 1:end - 1].float()
            targets = torch.tensor(ids[start:end], device=logits.device)
            out.append(torch.log_softmax(logits, -1).gather(1, targets[:, None]).sum().item())
    return out


def free_memory() -> None:
    gc.collect()
    torch.cuda.empty_cache()

# %% [markdown]
# ## 1. Tiny check: does GPTQModel read LinguaPrune's files exactly?
#
# ---
#
# 🇷🇺 **1. Маленькая проверка: читает ли GPTQModel файлы LinguaPrune правильно?**

# %%
def tiny_check() -> dict:
    from transformers import AutoModelForCausalLM, Qwen3MoeConfig

    from linguaprune.export import GPTQCheckpointWriter
    from linguaprune.gptq import make_quantization_prepare
    from linguaprune.streaming import ScoredSequence, StreamedModel, stream_evaluate

    root = DOWNLOAD_ROOT / "tiny"
    shutil.rmtree(root, ignore_errors=True)
    config = Qwen3MoeConfig(vocab_size=256, hidden_size=128, intermediate_size=128, moe_intermediate_size=128,
                            num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2, head_dim=32,
                            num_experts=8, num_experts_per_tok=2, max_position_embeddings=128, pad_token_id=0,
                            norm_topk_prob=True, decoder_sparse_step=1)
    torch.manual_seed(SEED)
    AutoModelForCausalLM.from_config(config, dtype=torch.float16).save_pretrained(root / "original")
    gen = torch.Generator().manual_seed(1)
    exam = [ScoredSequence(ids=tuple(torch.randint(1, 256, (n,), generator=gen).tolist()), span=(1, n))
            for n in (17, 33, 64)]
    calibration = [ScoredSequence(ids=tuple(torch.randint(1, 256, (64,), generator=gen).tolist())) for _ in range(8)]
    bits = np.array([[4, 3, 4, 3, 4, 4, 3, 4], [3, 4, 3, 4, 3, 4, 4, 3]])

    streamed = StreamedModel(root / "original", device="cuda:0", dtype=torch.float32)
    writer = GPTQCheckpointWriter(root / "packed", streamed.reader, group_size=128, default_bits=4,
                                  attention_bits=8, description={"name": "tiny"})
    prepare, _ = make_quantization_prepare(
        streamed, {"q": {"method": "gptq", "bits": bits}}, group_size=128, attention_bits=8,
        log=lambda _: None, on_quantized=lambda c, i, layer, rec: writer.write_layer(c, i, layer, rec))
    stream_evaluate(streamed, exam, ["q"], prepare=prepare, calibration={"q": calibration}, log=lambda _: None)
    writer.finish(num_layers=2)
    convert_to_gptq_v2(root / "packed")
    reference = [s.span_logprob for s in stream_evaluate(
        StreamedModel(root / "packed", device="cuda:0", dtype=torch.float32), exam, ["file"],
        log=lambda _: None)["file"]]

    model = GPTQModel.load(str(root / "packed"), device="cuda:0", backend=BACKEND.TORCH, dtype=torch.float16)
    runtime = span_logprobs(model.model, [(list(s.ids), s.span) for s in exam])
    del model
    free_memory()
    tokens = sum(s.span[1] - s.span[0] for s in exam)
    gap = abs(sum(runtime) - sum(reference)) / tokens  # nats per token
    return {"reference": reference, "runtime": runtime, "mean_abs_gap_nats_per_token": gap, "ok": gap < 0.02}


try:
    summary["tiny_check"] = tiny_check()
except Exception:
    summary["tiny_check"] = {"ok": False, "error": traceback.format_exc()}
save_summary()
print(json.dumps(summary["tiny_check"], indent=2)[:3000])
print("Tiny check passed:", summary["tiny_check"]["ok"])

# %% [markdown]
# ## 2. The real models from Hugging Face
#
# ---
#
# 🇷🇺 **2. Настоящие модели с Hugging Face**

# %%
scores_07 = pd.read_csv(SCORES_07)
check_rows = []  # (lang, ids, span) for the first N_CHECK Belebele passages per language, exactly as in notebook 07
for lang, df in load_eval_rows(EVAL_LANGUAGES, N_EVAL, SEED, reference_lang="en").items():
    if lang == "en":
        continue
    for position, (_, row) in enumerate(df.iterrows()):
        if position == N_CHECK:
            break
        check_rows.append((lang, position, row))


def reference_rows(config: str, lang: str) -> pd.DataFrame:
    rows = scores_07[(scores_07.config == config) & (scores_07.task == "belebele") & (scores_07.lang == lang)]
    return rows.reset_index(drop=True).iloc[:N_CHECK]


def evaluate_model(config: str, repo: str) -> dict:
    from huggingface_hub import snapshot_download

    result = {"repo": f"{HF_USER}/{repo}"}
    started = time.time()
    path = snapshot_download(f"{HF_USER}/{repo}", token=TOKEN, local_dir=DOWNLOAD_ROOT / repo)
    result["download_min"] = (time.time() - started) / 60
    result["modules_converted_to_v2"] = convert_to_gptq_v2(path)
    tokenizer = AutoTokenizer.from_pretrained(path)

    started = time.time()
    model = GPTQModel.load(path, device_map="auto", backend=BACKEND.TORCH, dtype=torch.float16)
    result["load_min"] = (time.time() - started) / 60
    result["gpu_memory_gb"] = [torch.cuda.memory_allocated(i) / 1e9 for i in range(torch.cuda.device_count())]
    print(f"{config}: loaded in {result['load_min']:.1f} min, GPU memory {result['gpu_memory_gb']}")

    # perplexity check against notebook 07 on the same passages
    sequences, per_lang = [], {}
    for lang, position, row in check_rows:
        ex = build_example(tokenizer, row)
        sequences.append((ex["ids"][0].tolist(), ex["passage_span"]))
        per_lang.setdefault(lang, []).append(len(sequences) - 1)
    runtime = span_logprobs(model.model, sequences)
    check = {}
    for lang, indices in per_lang.items():
        ref = reference_rows(config, lang)
        orig = reference_rows("original", lang)
        spans = [sequences[i][1][1] - sequences[i][1][0] for i in indices]
        if spans != ref.tokens.astype(int).tolist():
            raise RuntimeError(f"{lang}: passages differ from notebook 07 (token counts do not match)")
        tokens = sum(spans)
        ppl_runtime = float(np.exp(-sum(runtime[i] for i in indices) / tokens))
        ppl_07 = float(np.exp(-ref.logprob.sum() / tokens))
        ppl_original = float(np.exp(-orig.logprob.sum() / tokens))
        check[lang] = {"ppl_runtime": ppl_runtime, "ppl_notebook_07": ppl_07, "ppl_original": ppl_original,
                       "mismatch": ppl_runtime / ppl_07 - 1, "growth_vs_original": ppl_runtime / ppl_original - 1}
    result["perplexity_check"] = check
    result["reads_file_correctly"] = all(abs(c["mismatch"]) <= MAX_PPL_MISMATCH for c in check.values())
    print(pd.DataFrame(check).T.round(4))

    # questions
    answers = []
    for lang, question in QUESTIONS:
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": question}],
                                               tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(prompt, return_tensors="pt").to("cuda:0")
        started = time.time()
        with torch.inference_mode():
            output = model.model.generate(**inputs, max_new_tokens=MAX_NEW_TOKENS, do_sample=False)
        seconds = time.time() - started
        new = output[0, inputs.input_ids.shape[1]:]
        text = tokenizer.decode(new, skip_special_tokens=True).strip()
        answers.append({"lang": lang, "question": question, "answer": text, "tokens": len(new),
                        "tokens_per_second": len(new) / seconds})
        print(f"\n[{lang}] {question}\n→ {text}\n({len(new)} tokens, {len(new) / seconds:.1f} tok/s)")
    result["answers"] = answers
    del model
    free_memory()
    shutil.rmtree(DOWNLOAD_ROOT / repo, ignore_errors=True)
    return result


if not TOKEN:
    print("No HF_TOKEN: the real models are private, so this part is skipped.")
elif not summary["tiny_check"]["ok"]:
    print("The tiny check failed, so the real models are not downloaded. See the error above.")
else:
    for config, repo in MODELS.items():
        try:
            summary["models"][config] = evaluate_model(config, repo)
        except Exception:
            summary["models"][config] = {"repo": f"{HF_USER}/{repo}", "error": traceback.format_exc()}
            print(f"{config}: FAILED\n{summary['models'][config]['error']}")
            free_memory()
        save_summary()

# %% [markdown]
# ## 3. Verdict
#
# ---
#
# 🇷🇺 **3. Итог**

# %%
print("Tiny check:", "passed" if summary["tiny_check"]["ok"] else "FAILED")
for config, result in summary["models"].items():
    if "error" in result:
        print(f"{config}: could not run ({result['error'].strip().splitlines()[-1]})")
        continue
    speed = np.mean([a["tokens_per_second"] for a in result["answers"]])
    growth = {lang: f"{100 * c['growth_vs_original']:+.1f}%" for lang, c in result["perplexity_check"].items()}
    print(f"{config}: reads the file correctly = {result['reads_file_correctly']}, "
          f"perplexity vs. original {growth}, {speed:.1f} tokens/s")
save_summary()
print("Done:", sorted(p.name for p in OUTPUT_DIR.iterdir()))
