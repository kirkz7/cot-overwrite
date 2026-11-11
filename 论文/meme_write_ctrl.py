"""Controls for the write upper bound (PREREG_MEME_WRITE.md, section E; designed after seeing w3, written before any control output).

Why: `w3` (flag only) reached Cas 93.9 without thinking, where the evidence sessions themselves give 0. Three things differ between those
two contexts: the flag line, the selection of the target's chain, and the form (templated fact lines instead of conversation). Each
control removes one of them. Oracle annotations are used throughout: diagnostic, not a method.

  w2     w3 without the NEEDS RECHECK lines (one variable against w3: the flag)
  w2all  w2 + every other annotated fact of the episode as a plain line (one variable against w2: selection)
  wraw   every annotated fact of the episode, as stated, in order; no "changed from / was previously" lines, no flag, no selection
         (what a perfect extract-only writer would hold)

Rows go to the same files as the other write products (tag `w`), so `meme_write.py table` and `meme_reader.py judge` pick them up.

  python meme_write_ctrl.py dry
  python meme_write_ctrl.py answer [--think]
"""
import sys

import numpy as np

import meme_pilot as M
import meme_reader as R
import meme_write as W

CONDS = ("w2", "w2all", "wraw")
OTHER = "\n\nOther facts:\n"


def raw_lines(F, skip=()):
    return "\n".join(f"- {' '.join(g['fact_text'].split())}" for g in F if g["entity"] not in skip)


def products(ep):
    """{(phase, cond, entity): (question, context)}; every control has its own before-phase context"""
    out = {}
    bpos, apos = ep["before_questions"]["position_after_session"], ep["after_questions"]["position_after_session"]
    Fb, Fa = W.facts(ep, bpos), W.facts(ep, apos)
    before = {q["entity"][0]: q for q in ep["before_questions"]["questions"] if q["task_type"] in W.DEP_TASKS}
    for q in M.questions(ep):
        e, b = q["entity"][0], before[q["entity"][0]]
        out["before", "wraw", e] = (b, "Related information:\n" + raw_lines(Fb))
        out["after", "wraw", e] = (q, "Related information:\n" + raw_lines(Fa))
        if q["task_type"] == "Del":
            continue
        on_chain = {ep["root"]} | {x for x, _ in W.chain(ep, e)}
        w2 = "\n".join(ln for ln in W.flag_after(ep, q, Fa).split("\n") if not ln.startswith("- NEEDS RECHECK"))
        sb = W.state_before(ep, b, Fb)
        out["before", "w2", e], out["after", "w2", e] = (b, sb), (q, w2)
        out["before", "w2all", e] = (b, sb + OTHER + raw_lines(Fb, on_chain))
        out["after", "w2all", e] = (q, w2 + OTHER + raw_lines(Fa, on_chain))
    return out


def dry():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M.EMB)
    n = {}
    for ep in M.load():
        P, Pw = products(ep), W.products(ep)
        for (phase, cond, e), (q, c) in P.items():
            n.setdefault((phase, cond, q["task_type"]), []).append(len(tok(c, add_special_tokens=False).input_ids))
            if phase == "after" and cond == "w2":                 # exactly w3 minus its flag lines
                w3 = Pw["after", "w3", e][1].split("\n")
                assert [ln for ln in w3 if not ln.startswith("- NEEDS RECHECK")] == c.split("\n") and len(w3) - len(c.split("\n")) == q["hop"]
            if phase == "after" and cond == "w2all":
                assert c.startswith(P["after", "w2", e][1] + OTHER)
            if phase == "after" and q["task_type"] == "Cas":      # the new value is reachable only through a rule line, never stated as current
                assert q["gold_answer"] in c and f"current {W.name(e)}: {q['gold_answer']}" not in c, (ep["episode_id"], e, cond)
            assert "NEEDS RECHECK" not in c
        n.setdefault(("facts", "per episode", ""), []).append(len(W.facts(ep, ep["after_questions"]["position_after_session"])))
    for k, v in sorted(n.items()):
        print(k, "n", len(v), "median", int(np.median(v)), "min", int(min(v)), "max", int(max(v)))


if __name__ == "__main__":
    if sys.argv[1] == "dry":
        dry()
    else:
        import ctypes
        import os
        if os.name == "nt":                                       # process-scoped keep-awake, released on exit
            ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
        conds = (W.arg("conds") or ",".join(CONDS)).split(",")
        W.run_rows("w", "--think" in sys.argv, W.jobs_for(M.load(), conds, products))
