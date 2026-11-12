"""Hero figure for notebook 06: Kazakh/Russian perplexity growth vs. memory on Qwen3-30B-A3B.

Usage: python scripts/plot_qwen3_result.py  (reads experiments/06/full/summary.json)
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
SUMMARY = ROOT / "experiments/06/full/summary.json"
TARGET_GROWTH = 2.0  # percent, fixed before the run
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#1f1f1e", "#6b6a64", "#e6e5e0"
KK, RU = "#2a78d6", "#eb6834"  # categorical slots 1 and 2

TEXT = {
    "en": {
        "title": "Qwen3-30B-A3B compressed 4×: Kazakh and Russian stay within 2%",
        "xlabel": "Perplexity growth vs. the original 61 GB model (%, lower is better)",
        "target": "target ≤ +2%",
        "kk": "Kazakh (kk)",
        "ru": "Russian (ru)",
        "broken": "but KazMMLU falls to chance (26%)",
        "names": {
            "gptq 4-bit (kk+ru calib)": "uniform 4-bit",
            "gptq kkru 4/3 @80%": "kk+ru 4/3-bit, 80%",
            "gptq kkru 4/3 @90%": "kk+ru 4/3-bit, 90%",
            "gptq en 4/3 @90% (control)": "English 4/3-bit (control)",
            "gptq kkru 4/3/2 @80/95%": "kk+ru 4/3/2-bit",
        },
        "gb": "GB",
        "out": ROOT / "assets/qwen3_result.png",
    },
    "ru": {
        "title": "Qwen3-30B-A3B сжата в 4 раза: казахский и русский в пределах 2%",
        "xlabel": "Рост перплексии относительно исходной модели 61 ГБ (%, меньше — лучше)",
        "target": "цель ≤ +2%",
        "kk": "казахский (kk)",
        "ru": "русский (ru)",
        "broken": "но KazMMLU падает до угадывания (26%)",
        "names": {
            "gptq 4-bit (kk+ru calib)": "равномерно 4 бита",
            "gptq kkru 4/3 @80%": "kk+ru 4/3 бита, 80%",
            "gptq kkru 4/3 @90%": "kk+ru 4/3 бита, 90%",
            "gptq en 4/3 @90% (control)": "английский 4/3 (контроль)",
            "gptq kkru 4/3/2 @80/95%": "kk+ru 4/3/2 бита",
        },
        "gb": "ГБ",
        "out": ROOT / "assets/ru/qwen3_result.png",
    },
}


def plot(rows: list[dict], lang: str) -> None:
    t = TEXT[lang]
    rows = [r for r in rows if r["config"] in t["names"]][::-1]
    labels = [f'{t["names"][r["config"]]}\n{r["memory_gb"]:.1f} {t["gb"]} · {r["compression"]:.1f}×' for r in rows]
    fig, ax = plt.subplots(figsize=(9, 5.2), dpi=160, facecolor=SURFACE)
    ax.set_facecolor(SURFACE)
    height = 0.36
    for offset, key, color, name in ((height / 2, "kk", KK, t["kk"]), (-height / 2, "ru", RU, t["ru"])):
        values = [100 * r[key] for r in rows]
        ys = [i + offset for i in range(len(rows))]
        ax.barh(ys, values, height=height - 0.04, color=color, label=name)
        for y, v in zip(ys, values):
            ax.text(v + (0.25 if v >= 0 else -0.25), y, f"{v:+.1f}%", va="center",
                    ha="left" if v >= 0 else "right", fontsize=8.5, color=INK)
    ax.axvline(TARGET_GROWTH, color=MUTED, linestyle="--", linewidth=1)
    ax.text(TARGET_GROWTH + 0.2, len(rows) - 0.45, t["target"], color=MUTED, fontsize=9)
    ax.axvline(0, color=MUTED, linewidth=0.8)
    ax.set_yticks(range(len(rows)), labels, fontsize=9, color=INK)
    ax.set_xlabel(t["xlabel"], fontsize=9.5, color=INK)
    ax.set_title(t["title"], fontsize=11.5, color=INK, loc="left", pad=12)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    for side in ("top", "right", "left"):
        ax.spines[side].set_visible(False)
    ax.spines["bottom"].set_color(GRID)
    ax.tick_params(colors=MUTED, length=0)
    ax.legend(loc="upper right", frameon=False, fontsize=9)
    ax.set_xlim(-3.2, 17)
    broken = [i for i, r in enumerate(rows) if r["kazmmlu_kk"] < 0.4]
    for i in broken:
        ax.text(3.6, i + height / 2, t["broken"], va="center", fontsize=8.5, color=INK, style="italic")
    fig.tight_layout()
    t["out"].parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(t["out"], facecolor=SURFACE)
    plt.close(fig)
    print("wrote", t["out"].relative_to(ROOT))


if __name__ == "__main__":
    table = json.loads(SUMMARY.read_text())["table"]
    for language in TEXT:
        plot(table, language)
