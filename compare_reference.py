"""P0 port check (CLOUD_PLAN.md section 4.5): compare a fresh run with the desktop reference, row by row.
Pre-registered pass rule: >= 95% of rows give the same answer, and every per-condition accuracy differs by <= 2 points.
usage: python compare_reference.py logs | e15     (reference in reference/desktop/, fresh run in results/)
"""
import sys

import pandas as pd

from app_common import load_jsonl

SPEC = {  # name -> (file, key columns, answer column, condition columns for the aggregate check)
    "logs": ("app_logs_Qwen3-4B.jsonl", ["seed", "fmt", "order"], "pred", ["fmt", "order"]),
    "e15": ("e15_rule_Qwen3-4B.jsonl", ["id", "order", "qtype"], "response", ["qtype", "order"]),
}


def main():
    name = sys.argv[1]
    fn, key, ans, cond = SPEC[name]
    ref = pd.DataFrame(load_jsonl(f"reference/desktop/{fn}")).set_index(key)
    new = pd.DataFrame(load_jsonl(f"results/{fn}")).set_index(key)
    ix = ref.index.intersection(new.index)
    print(f"{name}: reference {len(ref)} rows, fresh {len(new)} rows, matched {len(ix)}")
    if len(ix) < len(ref):
        print("  WARNING: fresh run incomplete or keys differ")
    a, b = (x.loc[ix, ans].fillna("<none>").astype(str).str.strip() for x in (ref, new))   # unparsed answers compare equal
    same = (a == b).mean() * 100
    print(f"  same answer: {same:.1f}%  (pass >= 95)")
    r, n = ref.loc[ix].reset_index(), new.loc[ix].reset_index()
    acc = pd.DataFrame({"ref": r.groupby(cond).correct.mean() * 100, "fresh": n.groupby(cond).correct.mean() * 100})
    acc["diff"] = acc.fresh - acc.ref
    print(acc.round(1).to_string())
    worst = acc["diff"].abs().max()
    print(f"  largest per-condition difference: {worst:.1f} points  (pass <= 2)")
    print("  PORT CHECK:", "PASS" if same >= 95 and worst <= 2 else "FAIL - find the cause before running 32B")


if __name__ == "__main__":
    main()
