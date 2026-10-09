"""Diagnosis only (10-08, user): why does relevance order cost 14-17 points on LongMemEval knowledge-update (KU)
questions in the official retrieval + reading pipeline, and why does E18.1 not fix it? LongMemEval is SEEN data;
aggregates only, no conversation, question or answer text is printed.
Already known (CPU, cloud outputs): 88% of KU questions have every evidence session in the top 5; nearly every lost item
is one of them, so the loss is not a retrieval failure - the same 5 sessions, only their order changes.
Stage `ranks` (CPU): rebuild the official BM25 top-5 (explore_lme_official.with_retrieval, same code and data as the
cloud) and record, per KU question, where the NEWEST evidence session sits relative to the older evidence session(s)
in the relevance order (presented first / last / between) and in the date order (always last); join with the cloud's
judged correctness (4B and 8B, base and E18.1, thinking on / off). Prediction of "later in the prompt = newer":
under relevance order the losses concentrate where the newest evidence is presented BEFORE the older one.
usage: python diag_lme_rag.py ranks
"""
import argparse
import copy
import json
import os

import pandas as pd

OUT = "results/diag_lme_rag_ranks.jsonl"
MODELS = ["Qwen3-4B~vllm", "Qwen3-4B~vllm+think", "Qwen3-4B-e181~vllm+think", "Qwen3-8B-bf16~vllm", "Qwen3-8B-bf16~vllm+think",
          "Qwen3-8B-e181~vllm~pp1.5+think"]


def ranks(args):
    from explore_lme_official import load_s, with_retrieval, TOPK
    rows = []
    for e in load_s():
        if e["question_type"] != "knowledge-update":
            continue
        e2, rec = with_retrieval(copy.deepcopy(e))
        top = e2["retrieval_results"]["ranked_items"][:TOPK]
        ev = [(k, it["timestamp"]) for k, it in enumerate(top) if "answer" in it["corpus_id"]]   # (relevance rank, date)
        r = dict(question_id=e["question_id"], n_ev_top5=len(ev), recall_all=rec["recall_all"])
        if len(ev) >= 2:
            newest = max(ev, key=lambda x: x[1])
            older = [k for k, t in ev if (k, t) != newest]
            r.update(newest_rank=newest[0], older_ranks=older,
                     newest_pos=("first" if newest[0] < min(older) else "last" if newest[0] > max(older) else "between"),
                     newest_is_top1=newest[0] == 0)
        rows.append(r)
    with open(OUT, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    d = pd.DataFrame(rows).set_index("question_id")
    print("KU questions", len(d), "| >=2 evidence sessions in top 5:", int((d.n_ev_top5 >= 2).sum()),
          "| newest evidence presented first / between / last (relevance order):", d.newest_pos.value_counts().to_dict())
    from app_common import load_jsonl
    for m in MODELS:
        out = {}
        for order in ("date", "relevance"):
            p = f"results/cloud/lme_official_{m}~{order}_judged.jsonl"
            if os.path.exists(p):
                j = pd.DataFrame(load_jsonl(p)).set_index("question_id")
                out[order] = j.correct.fillna(False).astype(bool)
        if len(out) < 2:
            continue
        x = d.join(pd.DataFrame(out), how="inner")
        x = x[x.n_ev_top5 >= 2]
        tab = x.groupby("newest_pos")[["date", "relevance"]].mean().mul(100).round(1)
        tab["n"] = x.groupby("newest_pos").size()
        tab["lost"] = x.groupby("newest_pos").apply(lambda g: int((g["date"] & ~g["relevance"]).sum()))
        print("=" * 6, m); print(tab.to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["ranks"])
    args = ap.parse_args()
    {"ranks": ranks}[args.stage](args)


if __name__ == "__main__":
    main()
