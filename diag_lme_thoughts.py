"""Diagnosis only (10-09): E18.1's thoughts on LongMemEval official RAG, knowledge-update questions (diag_lme_rag.py).
SEEN data, aggregates only: nothing of the conversations, questions, answers or thoughts is printed. Dates in a thought are
found with a regex and compared with the session dates of the evidence (from the data file); no text is shown.
usage: python diag_lme_thoughts.py
"""
import datetime as dt
import json
import re

import pandas as pd

from app_common import load_jsonl
from explore_lme_official import load_s

FILES = {"E18.1 relevance": "results/lme_official_Qwen3-4B+e181-dec+think~relevance_judged.jsonl",
         "E18.1 date": "results/lme_official_Qwen3-4B+e181-dec+think~date_judged.jsonl",
         "base relevance": "results/lme_official_Qwen3-4B+think~relevance_judged.jsonl",
         "E18 relevance": "results/lme_official_Qwen3-4B+e18-dec+think~relevance_judged.jsonl",   # 10-09 test
         "E18 date": "results/lme_official_Qwen3-4B+e18-dec+think~date_judged.jsonl"}
DATE = re.compile(r"(\d{4})[/-](\d{1,2})[/-](\d{1,2})")


def ev_dates(e):
    """dates (date objects) of the evidence sessions, newest last"""
    ds = [d for sid, d in zip(e["haystack_session_ids"], e["haystack_dates"]) if sid in set(e["answer_session_ids"])]
    return sorted({dt.date(*map(int, DATE.search(d).groups())) for d in ds if DATE.search(d)})


def main():
    data = {e["question_id"]: e for e in load_s() if e["question_type"] == "knowledge-update"}
    ranks = pd.DataFrame(load_jsonl("results/diag_lme_rag_ranks.jsonl")).set_index("question_id")
    for name, path in FILES.items():
        if not load_jsonl(path):
            print(name, "- no results yet")
            continue
        d = pd.DataFrame(load_jsonl(path)).set_index("question_id").join(ranks[["newest_pos"]], how="inner")
        rows = []
        for qid, r in d.iterrows():
            full = r.get("full") or ""
            th = full.split("</think>")[0]
            evd = ev_dates(data[qid])
            found = [dt.date(*map(int, m)) for m in DATE.findall(th) if 1 <= int(m[1]) <= 12 and 1 <= int(m[2]) <= 31]
            listed = re.findall(r"^\s*-\s*(\d{4}[/-]\d{1,2}[/-]\d{1,2})", th, flags=re.M)
            ld = [dt.date(*map(int, DATE.search(x).groups())) for x in listed]
            rows.append(dict(
                qid=qid, pos=r.newest_pos, correct=bool(r.correct), finished=bool(r.finished),
                template=th.lstrip().startswith("<think>\nRecords about") or "from oldest to newest" in th,
                n_list=len(ld),
                has_newest=bool(evd) and evd[-1] in found,
                has_older=len(evd) > 1 and any(x in found for x in evd[:-1]),
                list_sorted=len(ld) >= 2 and ld == sorted(ld),
                list_has_both=len(evd) > 1 and evd[-1] in ld and any(x in ld for x in evd[:-1]),
                newest_last_in_list=bool(ld) and bool(evd) and ld[-1] == evd[-1]))
        x = pd.DataFrame(rows)
        x = x[x.pos.notna()]
        print("=" * 8, name, "| n", len(x), "| acc newest-first", round(x[x.pos == "first"].correct.mean() * 100, 1),
              "| newest-last", round(x[x.pos == "last"].correct.mean() * 100, 1))
        cols = ["template", "has_newest", "has_older", "list_has_both", "list_sorted", "newest_last_in_list", "finished"]
        t = x.groupby(["pos", "correct"])[cols].mean().mul(100).round(0)
        t["n"] = x.groupby(["pos", "correct"]).size()
        t["med_list"] = x.groupby(["pos", "correct"]).n_list.median()
        print(t.to_string())


if __name__ == "__main__":
    main()
