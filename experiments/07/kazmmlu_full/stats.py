"""Statistics of the full KazMMLU run (22,889 questions): accuracies, paired differences against the original, paired bootstrap 95% CI
(seed 0, 10,000 resamples) and exact two-sided McNemar p. Reads the three scores_*.csv written by the Kaggle runs. Prints a JSON table."""
import csv, json, sys
from math import comb
from pathlib import Path
import numpy as np

D = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
files = {"original": "scores_original.csv", "4/3-bit": "scores_4-3bit.csv", "4-bit": "scores_4bit.csv"}
rows = {k: list(csv.DictReader(open(D / v))) for k, v in files.items()}
n = {k: len(v) for k, v in rows.items()}; assert len(set(n.values())) == 1, n
out = {"n_total": n["original"]}
for lang in ("kk", "ru"):
    base = np.array([r["correct"] == "True" for r in rows["original"] if r["lang"] == lang])
    out[f"n_{lang}"] = int(base.size); out[f"original_{lang}"] = round(100 * base.mean(), 2)
    for name in ("4/3-bit", "4-bit"):
        other = np.array([r["correct"] == "True" for r in rows[name] if r["lang"] == lang])
        diff = other.astype(int) - base.astype(int)
        rng = np.random.default_rng(0)
        boots = np.sort([rng.choice(diff, diff.size).mean() * 100 for _ in range(10000)])
        b, c = int(((base) & (~other)).sum()), int(((~base) & other).sum())
        k, m = min(b, c), b + c
        p = min(1.0, 2 * sum(comb(m, i) for i in range(k + 1)) / 2 ** m)
        out[f"{name}_{lang}"] = {"acc": round(100 * other.mean(), 2), "diff": round(100 * diff.mean(), 2),
                                 "ci": [round(boots[250], 2), round(boots[9750], 2)], "mcnemar_p": p, "discordant": [b, c]}
print(json.dumps(out, indent=1))
