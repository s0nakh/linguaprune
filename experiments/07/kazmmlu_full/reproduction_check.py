"""Checks that the full KazMMLU run reproduces notebook 07 on the 762 questions of the earlier sample (questions matched by text and options), and that a difference, if any, is a near-tie (small gap between the top-2 answer scores). Needs the cached MBZUAI/KazMMLU dataset; run from this folder."""
import sys
import datasets, numpy as np, pandas as pd

D = "."; OLD = "../full/scores.csv"  # run from this folder
LET = "ABCDE"; cfg = {"original": "original", "4-3bit": "kk+ru 4/3-bit @90%", "4bit": "kk+ru 4-bit"}

def all_rows(lang):
    tag = "kaz" if lang == "kk" else "rus"
    subs = [s for s in datasets.get_dataset_config_names("MBZUAI/KazMMLU") if s.endswith(f"in {tag})")]
    return pd.concat([datasets.load_dataset("MBZUAI/KazMMLU", s, split="test").to_pandas().assign(subset_name=s) for s in subs], ignore_index=True)

def sample(lang, n=400, seed=0):  # copy of linguaprune.exam._kazmmlu_rows
    pool = all_rows(lang); per = max(1, n // pool.Subset.nunique())
    parts = [g.sample(n=min(per, len(g)), random_state=seed) for _, g in pool.groupby("Subset")]
    return pd.concat(parts, ignore_index=True).sample(n=min(n, sum(len(p) for p in parts)), random_state=seed).reset_index(drop=True)

# map (lang, question) -> row index in the new file (rows with a valid answer key, in all_rows order)
meta = {}
for lang in ("kk", "ru"):
    pool = all_rows(lang); keep = []
    for _, r in pool.iterrows():
        opts = [l for l in LET if str(r[f"Option {l}"]) not in ("None", "nan", "")]
        if r["Answer Key"] in opts: keep.append((lang, r["Question"].strip(), tuple(str(r[f"Option {l}"]) for l in opts), r["Answer Key"]))
    meta[lang] = keep
order = meta["kk"] + meta["ru"]
index = {k: i for i, k in enumerate(order)}
expected = []
for lang in ("kk", "ru"):
    for _, r in sample(lang).iterrows():
        opts = [l for l in LET if str(r[f"Option {l}"]) not in ("None", "nan", "")]
        if r["Answer Key"] in opts: expected.append((lang, r["Question"].strip(), tuple(str(r[f"Option {l}"]) for l in opts), r["Answer Key"]))

old_all = pd.read_csv(OLD); old_all = old_all[old_all.task == "kazmmlu"]
for name, oldname in cfg.items():
    new = pd.read_csv(f"{D}/scores_{name}.csv"); assert len(new) == len(order)
    old = old_all[old_all.config == oldname].reset_index(drop=True); assert len(old) == len(expected)
    gaps_all = new[list(LET)].apply(lambda r: np.sort(r.dropna().values)[::-1][:2], axis=1)
    gap = np.array([g[0] - g[1] for g in gaps_all])
    diff = [i for i, (k, oc) in enumerate(zip(expected, old.choice)) if new.choice[index[k]] != oc]
    gd = gap[[index[expected[i]] for i in diff]]
    if not len(gd):
        print(f"{name}: 0 of {len(old)} differ"); continue
    print(f"{name}: {len(diff)} of {len(old)} differ; top-2 gap of differing questions: median {np.median(gd):.3f}, max {gd.max():.3f}, "
          f"all {len(gap)}: median {np.median(gap):.2f}, share of questions with gap<={gd.max():.2f}: {(gap <= gd.max()).mean():.3f}")
    print("   sorted gaps:", np.round(np.sort(gd), 3).tolist())


from collections import Counter
dups = sum(1 for v in Counter(order).values() if v > 1)
print("identical (question, options, answer) keys that occur twice in the pool:", dups)
for name, oldname in cfg.items():
    new = pd.read_csv(f"{D}/scores_{name}.csv"); old = old_all[old_all.config == oldname].reset_index(drop=True)
    ok = sum(int(new.choice[index[k]]) == int(c) for k, c in zip(expected, old.choice))
    print(f"{name}: identical answers on {ok} of {len(old)}")
