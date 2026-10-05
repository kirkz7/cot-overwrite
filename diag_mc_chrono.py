"""Diagnosis only (no scoring change): why does E13 lose accuracy on MemConflict chrono?
Aggregates only, no conversation or response text is printed (MemConflict is held-out).
usage: python diag_mc_chrono.py
"""
import re

import pandas as pd

from app_common import load_jsonl

MODELS = ["Qwen3-4B", "Qwen3-4B+e13-dec", "Qwen3-4B+e13-chr"]


def norm(s):
    return " ".join(re.findall(r"[a-z0-9]+", str(s).lower()))


def has(text, v):
    v = norm(v)
    return bool(v) and v in norm(text)


def load(name):
    raw = pd.DataFrame(load_jsonl(f"results/extmem_{name}.jsonl"))
    raw = raw[raw.task == "memconf"]
    j = pd.DataFrame(load_jsonl(f"results/extmem_{name}_judged.jsonl"))
    j = j[j.task == "memconf"][["id", "cond", "label"]]
    df = raw.merge(j, on=["id", "cond"])
    resp = df.response.astype(str)
    full = df["full"].astype(str) if "full" in df else resp
    df["has_answer_tag"] = full.str.contains("Answer:")
    df["full_len_words"] = full.str.split().str.len()
    df["resp_len_words"] = resp.str.split().str.len()
    df["new_in"] = [has(r, v) for r, v in zip(resp, df.new)]
    df["old_in"] = [has(r, v) for r, v in zip(resp, df.old)]
    df["str_class"] = ["both" if n and o else "new_only" if n else "old_only" if o else "neither"
                       for n, o in zip(df.new_in, df.old_in)]
    df["idk"] = resp.str.lower().str.contains(r"not (?:mention|specif|state|provid)|no (?:information|change)|unknown|cannot|does not")
    return df


def main():
    dfs = {m: load(m) for m in MODELS}
    print("columns:", sorted(dfs[MODELS[0]].columns))
    for m, df in dfs.items():
        c = df[df.cond == "chrono"]
        print("=" * 8, m, "chrono n =", len(c))
        print("  judge labels %:", (c.label.value_counts(normalize=True) * 100).round(1).to_dict())
        print("  string class %:", (c.str_class.value_counts(normalize=True) * 100).round(1).to_dict())
        print("  label x string class (counts):\n" + pd.crosstab(c.label, c.str_class).to_string())
        print(f"  'Answer:' present {c.has_answer_tag.mean() * 100:.1f}% | median words full {c.full_len_words.median()} "
              f"answer {c.resp_len_words.median()} | idk-like {c.idk.mean() * 100:.1f}%")
    base = dfs[MODELS[0]].query("cond == 'chrono'").set_index("id")
    for m in MODELS[1:]:
        e = dfs[m].query("cond == 'chrono'").set_index("id")
        ids = base.index.intersection(e.index)
        lost = [i for i in ids if base.label[i] == "A" and e.label[i] != "A"]
        gained = [i for i in ids if base.label[i] != "A" and e.label[i] == "A"]
        print("=" * 8, m, "vs base on chrono: lost", len(lost), "gained", len(gained), "of", len(ids))
        L = e.loc[lost]
        print("  lost -> E13 label:", L.label.value_counts().to_dict(), "| E13 string class:", L.str_class.value_counts().to_dict())
        print("  lost -> base string class:", base.loc[lost].str_class.value_counts().to_dict())
        print(f"  lost: 'Answer:' present {L.has_answer_tag.mean() * 100:.0f}% | idk-like {L.idk.mean() * 100:.0f}% | "
              f"median context tok {int(L.n_tok.median())} vs all {int(e.n_tok.median())}")
        # does the drop depend on context length or number of sessions?
        e2 = e.loc[ids].assign(base_A=base.loc[ids].label == "A", e_A=e.loc[ids].label == "A")
        e2["len_q"] = pd.qcut(e2.n_tok, 3, labels=["short", "mid", "long"])
        print("  by context-length tercile (base A%, E13 A%):\n" +
              (e2.groupby("len_q", observed=True)[["base_A", "e_A"]].mean() * 100).round(1).to_string())


if __name__ == "__main__":
    main()
