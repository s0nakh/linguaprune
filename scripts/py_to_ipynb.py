"""Convert a percent-format script (# %% cells) into a Kaggle-ready .ipynb.

Usage: python3 scripts/py_to_ipynb.py notebooks/01_expert_profiling.py
"""

import json
import sys
from pathlib import Path

CELL_MARKER = "# %%"
MARKDOWN_MARKER = "# %% [markdown]"
KAGGLE_METADATA = {"accelerator": "nvidiaTeslaT4", "isInternetEnabled": True, "isGpuEnabled": True}


def split_cells(source: str) -> list[tuple[str, list[str]]]:
    cells: list[tuple[str, list[str]]] = []
    for line in source.splitlines():
        if line.startswith(CELL_MARKER):
            kind = "markdown" if line.startswith(MARKDOWN_MARKER) else "code"
            cells = [*cells, (kind, [])]
        elif cells:
            kind, lines = cells[-1]
            cells = [*cells[:-1], (kind, [*lines, line])]
    return cells


def to_notebook_cell(kind: str, lines: list[str]) -> dict:
    trimmed = "\n".join(lines).strip("\n").splitlines()
    if kind == "markdown":
        trimmed = [line[2:] if line.startswith("# ") else line.lstrip("#") for line in trimmed]
    source = [f"{line}\n" for line in trimmed[:-1]] + trimmed[-1:]
    cell = {"cell_type": kind, "metadata": {}, "source": source}
    return {**cell, "execution_count": None, "outputs": []} if kind == "code" else cell


def convert(path: Path) -> Path:
    cells = [to_notebook_cell(kind, lines) for kind, lines in split_cells(path.read_text())]
    if not cells:
        raise ValueError(f"{path}: no '{CELL_MARKER}' cells found")
    notebook = {
        "cells": cells,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python"},
            "kaggle": KAGGLE_METADATA,
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    target = path.with_suffix(".ipynb")
    target.write_text(json.dumps(notebook, ensure_ascii=False, indent=1))
    return target


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit(__doc__)
    print(f"written {convert(Path(sys.argv[1]))}")
