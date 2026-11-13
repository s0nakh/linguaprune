"""Scoring GGUF files and driving the llama.cpp tools (importance matrix, quantization).

The scoring path needs `llama-cpp-python`; the tool helpers only build command lines and run them,
so they can be tested without llama.cpp installed.
"""

from __future__ import annotations

import os
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from linguaprune.exam import ExamItem

TOOL_TARGETS = ("llama-imatrix", "llama-quantize")
T4_CMAKE_FLAGS = (  # CUDA for the T4 only; no VMM (no libcuda at build time); nothing we do not run
    "-DGGML_CUDA=ON", "-DCMAKE_CUDA_ARCHITECTURES=75", "-DGGML_CUDA_NO_VMM=ON", "-DGGML_CUDA_NCCL=OFF",
    "-DGGML_CUDA_FA=OFF", "-DBUILD_SHARED_LIBS=OFF", "-DLLAMA_OPENSSL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
    "-DLLAMA_BUILD_EXAMPLES=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_BUILD_APP=OFF", "-DCMAKE_BUILD_TYPE=Release",
)
IMATRIX_CHUNK_TOKENS = 512
LOG_TAIL_LINES = 60


def log_softmax(rows: np.ndarray) -> np.ndarray:
    rows = rows.astype(np.float64)
    top = rows.max(axis=-1, keepdims=True)
    return rows - top - np.log(np.exp(rows - top).sum(axis=-1, keepdims=True))


def span_logprob(logits: np.ndarray, ids: Sequence[int], span: tuple[int, int]) -> float:
    """Sum of log-probabilities of tokens ids[start:end], each predicted from the logits one position earlier."""
    start, end = span
    if start < 1:
        raise ValueError("the scored span must start after the first token")
    rows = log_softmax(logits[start - 1:end - 1])
    return float(rows[np.arange(end - start), list(ids[start:end])].sum())


def score_items(llm, items: Sequence[ExamItem], keep: Sequence[int], config: str) -> list[dict]:
    """Score exam items with a `llama_cpp.Llama` created with `logits_all=True`; rows in notebook 07 format."""
    rows = []
    for i in keep:
        item = items[i]
        llm.reset()
        llm.eval(list(item.ids))
        logits = np.asarray(llm.scores[: len(item.ids)])
        choices = None if item.choice_ids is None else [float(logits[-1, c]) for c in item.choice_ids]
        rows.append({"config": config, **item.meta,
                     "logprob": None if item.span is None else span_logprob(logits, item.ids, item.span),
                     "tokens": None if item.span is None else item.span[1] - item.span[0],
                     "choice": None if choices is None else int(np.argmax(choices))})
    return rows


def score_gguf(path: Path, items: Sequence[ExamItem], keep: Sequence[int], config: str, n_ctx: int) -> tuple[list[dict], dict]:
    from llama_cpp import Llama

    started = time.time()
    llm = Llama(model_path=str(path), n_gpu_layers=-1, n_ctx=n_ctx, n_batch=512, logits_all=True, verbose=False)
    load_min = (time.time() - started) / 60
    started = time.time()
    try:
        rows = score_items(llm, items, keep, config)
    finally:
        del llm
    return rows, {"file_gb": path.stat().st_size / 1e9, "load_min": load_min, "score_min": (time.time() - started) / 60}


@dataclass(frozen=True)
class Tools:
    imatrix: Path
    quantize: Path


def run_logged(command: Sequence[str], log_path: Path, env: dict | None = None) -> float:
    """Run a command, keep its full output in `log_path`; on failure show the last lines. Returns minutes."""
    started = time.time()
    with open(log_path, "w") as log:
        result = subprocess.run([str(c) for c in command], stdout=log, stderr=subprocess.STDOUT, env=env)
    if result.returncode != 0:
        tail = Path(log_path).read_text(errors="replace").splitlines()[-LOG_TAIL_LINES:]
        print("\n".join(tail))
        raise RuntimeError(f"{Path(command[0]).name} failed with code {result.returncode}; log: {log_path}")
    return (time.time() - started) / 60


def build_tools(source: Path, build_dir: Path, nvcc: str, log_path: Path) -> Tools:
    """Compile llama-imatrix and llama-quantize (static, CUDA for T4) from a llama.cpp source tree."""
    env = {**os.environ, "CUDACXX": nvcc, "PATH": f"{os.path.dirname(nvcc)}:{os.environ['PATH']}"}
    run_logged(["cmake", "-S", source, "-B", build_dir, *T4_CMAKE_FLAGS], log_path, env)
    run_logged(["cmake", "--build", build_dir, "--target", *TOOL_TARGETS, "-j", str(os.cpu_count() or 4)],
               log_path.with_suffix(".build.log"), env)
    return Tools(imatrix=Path(build_dir) / "bin" / "llama-imatrix", quantize=Path(build_dir) / "bin" / "llama-quantize")


def imatrix_command(tool: Path, model: Path, text: Path, out: Path, chunks: int | None = None) -> list[str]:
    """Collect an importance matrix; `--fit on` lets llama.cpp keep on the CPU what does not fit on the GPUs."""
    command = [str(tool), "-m", str(model), "-f", str(text), "-o", str(out), "-c", str(IMATRIX_CHUNK_TOKENS),
               "-b", str(IMATRIX_CHUNK_TOKENS), "--fit", "on", "-fa", "off", "--no-ppl"]
    if chunks is not None:
        command += ["--chunks", str(chunks)]
    return command


def merge_imatrix_command(tool: Path, inputs: Sequence[Path], out: Path) -> list[str]:
    """Combine importance matrices (statistics are summed, weighted by how many tokens each saw)."""
    if len(inputs) < 2:
        raise ValueError("merging needs at least two importance matrices")
    command = [str(tool)]
    for path in inputs:
        command += ["--in-file", str(path)]
    return command + ["-o", str(out)]


def quantize_command(tool: Path, source: Path, out: Path, quant_type: str, imatrix: Path | None,
                     tensor_types: Sequence[str] = ()) -> list[str]:
    """Requantize `source` (e.g. Q8_0) to `quant_type`, optionally guided by an importance matrix."""
    command = [str(tool), "--allow-requantize"]
    if imatrix is not None:
        command += ["--imatrix", str(imatrix)]
    for override in tensor_types:
        if "=" not in override:
            raise ValueError(f"tensor type override must look like name=type, got {override!r}")
        command += ["--tensor-type", override]
    return command + [str(source), str(out), quant_type, str(os.cpu_count() or 4)]
