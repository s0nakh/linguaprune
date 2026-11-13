"""The fixed exam of notebooks 07 and later: Belebele kk/ru/en, KazMMLU kk/ru and factual probes.

Every runtime (PyTorch, GPTQModel, llama.cpp) is scored on exactly these token ids, so results are
comparable item by item. `check_against_reference` proves that a rebuilt exam matches the scores
saved by notebook 07.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

EVAL_LANGUAGES = {"kk": "kaz_Cyrl", "ru": "rus_Cyrl", "en": "eng_Latn"}
N_EVAL, SEED, KAZMMLU_PER_LANGUAGE = 150, 0, 400  # identical to notebook 07
LETTERS = ("A", "B", "C", "D", "E")
FACT_PROBES = (
    ("kk", "Қазақстанның астанасы —", " Астана", (" Алматы", " Шымкент")),
    ("ru", "Столица Казахстана —", " Астана", (" Алматы", " Москва")),
    ("en", "The capital of Kazakhstan is", " Astana", (" Almaty", " Moscow")),
    ("kk", "Қазақстанның ұлттық валютасы —", " теңге", (" рубль", " доллар")),
    ("ru", "Национальная валюта Казахстана —", " тенге", (" рубль", " доллар")),
    ("en", "The national currency of Kazakhstan is the", " tenge", (" ruble", " dollar")),
    ("kk", "Абай Құнанбайұлы — қазақтың ұлы", " ақыны", (" футболшысы", " ғарышкері")),
    ("ru", "Абай Кунанбаев — великий казахский", " поэт", (" футболист", " космонавт")),
    ("en", "Abai Kunanbayev was a great Kazakh", " poet", (" footballer", " astronaut")),
)


@dataclass(frozen=True)
class ExamItem:
    """Token ids to score; `span` marks the tokens whose log-probability counts; `choice_ids` are answer letters."""

    ids: tuple[int, ...]
    span: tuple[int, int] | None
    choice_ids: tuple[int, ...] | None
    meta: dict


def _kazmmlu_rows(language: str, n: int, seed: int):
    import datasets
    import pandas as pd

    tag = "kaz" if language == "kk" else "rus"
    subsets = [s for s in datasets.get_dataset_config_names("MBZUAI/KazMMLU") if s.endswith(f"in {tag})")]
    pool = pd.concat([datasets.load_dataset("MBZUAI/KazMMLU", s, split="test").to_pandas() for s in subsets],
                     ignore_index=True)
    per_subject = max(1, n // pool.Subset.nunique())
    parts = [group.sample(n=min(per_subject, len(group)), random_state=seed) for _, group in pool.groupby("Subset")]
    sample = pd.concat(parts, ignore_index=True)
    return sample.sample(n=min(n, len(sample)), random_state=seed).reset_index(drop=True)


def build_exam(tokenizer) -> list[ExamItem]:
    """Rebuild the notebook 07 exam, in the same order, with the given (Qwen3) tokenizer."""
    from linguaprune.belebele import build_example, load_eval_rows

    def encode(text: str) -> tuple[int, ...]:
        return tuple(tokenizer(text, add_special_tokens=False).input_ids)

    letter_ids = [encode(" " + letter)[0] for letter in LETTERS]
    items = []
    for lang, df in load_eval_rows(EVAL_LANGUAGES, N_EVAL, SEED, reference_lang="en").items():
        for _, row in df.iterrows():
            ex = build_example(tokenizer, row)
            items.append(ExamItem(tuple(ex["ids"][0].tolist()), tuple(ex["passage_span"]), tuple(letter_ids[:4]),
                                  {"task": "belebele", "lang": lang, "label": ex["label"]}))
    for lang in ("kk", "ru"):
        for _, row in _kazmmlu_rows(lang, KAZMMLU_PER_LANGUAGE, SEED).iterrows():
            options = [(l, str(row[f"Option {l}"])) for l in LETTERS
                       if str(row[f"Option {l}"]) not in ("None", "nan", "")]
            keys = [o[0] for o in options]
            if row["Answer Key"] not in keys:
                continue
            text = f"Question: {row['Question'].strip()}" + "".join(f"\n{l}. {o}" for l, o in options) + "\nAnswer:"
            items.append(ExamItem(encode(text), None, tuple(letter_ids[LETTERS.index(l)] for l in keys),
                                  {"task": "kazmmlu", "lang": lang, "label": keys.index(row["Answer Key"])}))
    for probe, (lang, prompt, correct, wrong) in enumerate(FACT_PROBES):
        prompt_ids = encode(prompt)
        for rank, answer in enumerate((correct, *wrong)):
            ids = prompt_ids + encode(answer)
            items.append(ExamItem(ids, (len(prompt_ids), len(ids)), None,
                                  {"task": "fact", "lang": lang, "probe": probe, "candidate": rank}))
    return items


def check_against_reference(items: list[ExamItem], reference, n_ctx: int) -> None:
    """Raise if `items` do not line up row by row with a reference score table (task, language, span length)."""
    if len(reference) != len(items):
        raise RuntimeError(f"exam has {len(items)} items, the reference has {len(reference)}")
    for i, item in enumerate(items):
        if (reference.task.iloc[i], reference.lang.iloc[i]) != (item.meta["task"], item.meta["lang"]):
            raise RuntimeError(f"item {i}: task/language differ from the reference")
        if item.span is not None and int(reference.tokens.iloc[i]) != item.span[1] - item.span[0]:
            raise RuntimeError(f"item {i}: span length differs from the reference")
    longest = max(len(item.ids) for item in items)
    if longest > n_ctx:
        raise RuntimeError(f"an exam item has {longest} tokens, more than the context window ({n_ctx})")


def first_items_per_task(items: list[ExamItem], per_task: int) -> list[int]:
    """Indices of the first `per_task` items of every (task, language), in exam order (for dry runs)."""
    seen: dict[tuple[str, str], int] = {}
    keep = []
    for i, item in enumerate(items):
        key = (item.meta["task"], item.meta["lang"])
        if seen.get(key, 0) < per_task:
            keep.append(i)
        seen[key] = seen.get(key, 0) + 1
    return keep


def summarize(scores, size_gb: dict[str, float], baseline: str):
    """One row per config: perplexity and accuracy per language, growth vs. `baseline`, compression.

    `scores` has the notebook 07 columns: config, task, lang, label, logprob, tokens, choice, probe, candidate.
    """
    import pandas as pd

    rows = []
    for config, group in scores.groupby("config", sort=False):
        row = {"config": config, "size_gb": size_gb[config]}
        for lang in ("kk", "ru", "en"):
            spans = group[(group.task == "belebele") & (group.lang == lang)]
            row[f"ppl_{lang}"] = float(np.exp(-spans.logprob.sum() / spans.tokens.sum()))
            row[f"belebele_{lang}"] = float((spans.choice == spans.label).mean())
        for lang in ("kk", "ru"):
            mmlu = group[(group.task == "kazmmlu") & (group.lang == lang)]
            row[f"kazmmlu_{lang}"] = float((mmlu.choice == mmlu.label).mean())
        facts = group[group.task == "fact"]
        row["facts"] = float(np.mean([g.sort_values("candidate").logprob.to_numpy().argmax() == 0
                                      for _, g in facts.groupby("probe")]))
        rows.append(row)
    table = pd.DataFrame(rows).set_index("config")
    if baseline not in table.index:
        raise KeyError(f"baseline {baseline!r} is not among the configs")
    base = table.loc[baseline]
    growth = {f"growth_{lang}": table[f"ppl_{lang}"] / base[f"ppl_{lang}"] - 1 for lang in ("kk", "ru", "en")}
    return table.assign(**growth, compression=base.size_gb / table.size_gb).sort_values("size_gb", ascending=False)
