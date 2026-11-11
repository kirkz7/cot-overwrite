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

CONDS = ("w2", "w2all", "wraw", "w1", "w1f")
# w1 / w1f (section F, added after the first three controls): the "was previously" wording of w2 already says the value is old.
#   w1   the chain's lines as a state tracker without dependency awareness would hold them after the change: the root line is
#        updated ("changed from A to B"), the downstream entry still reads "User's current X: <old value>"; dependency and rule lines kept
#   w1f  w1 + the same NEEDS RECHECK lines as w3 (one variable against w1: the flag; against w3: the wording of the old value)
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
        for cond, flag in (("w1", False), ("w1f", True)):
            lines = [f"- User's {W.name(ep['root'])} changed from '{ep['root_change']['before']}' to '{ep['root_change']['after']}'."]
            for i, (x, src) in enumerate(W.chain(ep, e)):
                lines += W.entity_lines(ep, x, Fa, True)
                if flag:
                    lines.append(f"- NEEDS RECHECK: this value depends on {W.name(src)}, which " + ("changed after the value was stated." if i == 0 else "is flagged NEEDS RECHECK above."))
            out["before", cond, e], out["after", cond, e] = (b, sb), (q, "Related information:\n" + "\n".join(lines))
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
            assert ("NEEDS RECHECK" in c) == (cond == "w1f" and phase == "after")
            if phase == "after" and cond in ("w1", "w1f"):        # w1f = w3 with "current" instead of "was previously"; w1 = w1f minus the flags
                w3 = Pw["after", "w3", e][1]
                for x, _ in W.chain(ep, e):
                    w3 = w3.replace(f"- User's {W.name(x)} was previously '{ep['entities'][x]['before']}'.", f"- User's current {W.name(x)}: {ep['entities'][x]['before']}")
                assert P["after", "w1f", e][1] == w3 and "was previously" not in c
                assert [ln for ln in w3.split("\n") if not ln.startswith("- NEEDS RECHECK")] == P["after", "w1", e][1].split("\n")
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
