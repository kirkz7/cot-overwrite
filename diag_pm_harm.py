"""Diagnosis only: why do the trained readers (E13 / E17) score 11-16 points below the base model on PersonaMem?
Aggregates only, no conversation, option or response text is printed (PersonaMem is held-out).
Checks: output shape (reasoning list / "Answer:" / letter-only), letter bias vs gold, accuracy by context length,
and whether the trained reader's errors fall on items the base model gets right.
usage: python diag_pm_harm.py
"""
import numpy as np
import pandas as pd

from app_common import load_jsonl
from explore_longconv import pm_pred, personamem

MODELS = ["Qwen3-4B", "Qwen3-4B+e13-dec", "Qwen3-4B+e17-dec"]


def load(m, gold):
    a = pd.DataFrame([r for r in load_jsonl(f"results/longconv_{m}.jsonl") if r["task"] == "personamem"])
    full = a["full"].where(a["full"].notna(), a.response).astype(str) if "full" in a else a.response.astype(str)
    a["pred"] = full.map(pm_pred)
    a["gold"] = [gold[(i, c)] for i, c in zip(a.id, a.cond)]
    a["correct"] = a.pred == a.gold
    a["has_list"] = full.str.contains("from oldest to newest")
    a["has_answer"] = full.str.contains("Answer:")
    a["words"] = full.str.split().str.len()
    return a


def main():
    gold = {(x["id"], x["cond"]): x["gold"] for x in personamem()}
    d = {m: load(m, gold) for m in MODELS}
    for m, a in d.items():
        print("=" * 8, m, "n =", len(a), f"acc {a.correct.mean() * 100:.1f}")
        print(f"  output: dated list {a.has_list.mean() * 100:.0f}% | 'Answer:' {a.has_answer.mean() * 100:.0f}% | "
              f"median words {a.words.median():.0f} | unparsed {a.pred.isna().mean() * 100:.1f}%")
        print("  predicted letter %:", (a.pred.value_counts(normalize=True) * 100).round(1).sort_index().to_dict(),
              "| gold %:", (a.gold.value_counts(normalize=True) * 100).round(1).sort_index().to_dict())
        print("  acc by gold letter:", (a.groupby("gold").correct.mean() * 100).round(1).to_dict())
        if a.has_list.any():
            print("  acc with list / without:", round(a[a.has_list].correct.mean() * 100, 1), "/",
                  round(a[~a.has_list].correct.mean() * 100, 1), "| n with list", int(a.has_list.sum()))
    base = d["Qwen3-4B"].set_index(["id", "cond"])
    for m in MODELS[1:]:
        e = d[m].set_index(["id", "cond"])
        ix = base.index.intersection(e.index)
        b, x = base.loc[ix], e.loc[ix]
        print("=" * 8, m, "vs base: lost", int((b.correct & ~x.correct).sum()), "gained", int((~b.correct & x.correct).sum()),
              "of", len(ix))
        q = pd.qcut(b.n_tok, 3, labels=["short", "mid", "long"])
        print("  acc by context-length tercile (base, trained):",
              {k: (round(b.correct[q == k].mean() * 100, 1), round(x.correct[q == k].mean() * 100, 1)) for k in ["short", "mid", "long"]},
              "| token range", int(b.n_tok.min()), "-", int(b.n_tok.max()))
        lost = b.correct & ~x.correct
        print("  lost items: trained picked letter", x.pred[lost].value_counts().to_dict(), "| gold", b.gold[lost].value_counts().to_dict())
        print("  lost items: trained output had dated list", f"{x.has_list[lost].mean() * 100:.0f}%")


if __name__ == "__main__":
    main()
