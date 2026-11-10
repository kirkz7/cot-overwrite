"""Write-side upper bounds and a StateMem reimplementation on MEME (diagnosis only). Design, predictions and decision
lines: PREREG_MEME_WRITE.md (written first). Exploration episodes only; no conversation text is printed, aggregates only.
Reader, answer prompt, judge and the real-pass rule are those of meme_reader.py; judging is `meme_reader.py judge`.

usage: python 论文/meme_write.py dry | selftest
       python 论文/meme_write.py answer [--think] --conds=w4,w3,w3s | --conds=w4rag,w3rag
       python 论文/meme_write.py ingest [--writer=Qwen3-8B] [--eps=pl_001,...]      (StateMem write side, resumable)
       python 论文/meme_write.py smread [--think] [--writer=Qwen3-8B]               (StateMem read side)
       python 论文/meme_write.py check [--writer=Qwen3-8B] | table [--pairs=tag/cond:tag/cond,...]
"""
import json
import os
import re
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import meme_pilot as M
import meme_reader as R

PILOT = ["pl_001", "pl_002", "pl_003", "sw_001", "sw_002", "sw_003"]     # fixed in the pre-registration
ENC_GEN = 600
CONNECT = "\n\nOther conversation history:\n"        # verbatim from MEME golden_memory.py (--with-filler)
BEFORE_OF = {"w3": "w4", "w3rag": "w4rag"}           # w3 and w4 share the identical before-phase prompt
DEP_TASKS = ("Cas", "Abs", "Del")

# --- verbatim from StateMem (arXiv 2608.19652) Appendix E.3 and E.1; PDF line wraps joined, list indentation ours ---
RECOMPUTE_GUIDANCE = """When answering, work through each relevant piece of state. Specifically:
- Hard constraints (medical, scope-bound) take priority over recent soft preferences.
- A unit flagged NEEDS RECHECK is STALE: an input it was derived from has changed, so its baked-in value is now wrong. Do NOT use that number. RECOMPUTE it: take its stated derivation (e.g. 'X = A * 2'), substitute the CURRENT value of each input from the active state above, and evaluate. If the result feeds another NEEDS RECHECK unit, recompute that one from the value you just derived -- propagate through the chain in order before answering. Only recompute units on the path to the question; leave others.
- For cascade-coupled units, surface the downstream implication explicitly."""

ENC_PROMPT = """You are a state maintenance system for a task-executing agent. After each conversation turn, you update the agent's tracked state.

## Current conversation state (from prior turns):
<<current_state>>

## Latest turn (turn <<turn_number>>):
<<latest_turn>>

## Your job:
Analyze the latest turn and determine what state updates are needed. You can:

1. ADD new units: New constraints, commitments, or facts that emerged this turn.
   - From USER statements: explicit requests, preferences, corrections
   - From TOOL results: new facts revealed by tool calls (e.g., membership level, flight status, reservation details, payment info)

2. SUPERSEDE existing units: When new information contradicts or updates an existing unit.
   - Example: user changes payment preference from credit card to gift card
   - Example: tool result shows user is silver, not gold as claimed earlier
   - Include the ID of the unit being superseded.
   - Look ACTIVELY for supersession patterns: "actually", "instead", "no longer", "switched to", "moved off", "cancelled", "replaced X with Y", "now Y not X".

3. TRACK PROGRESS: If the user requested multiple actions (e.g., "downgrade all 5 reservations", "cancel these 3 flights"), track what has been completed and what remains.
   - Add a progress unit like: "Completed 2 of 5 reservation downgrades. Remaining: X7BYG1, EQ1G6C, BOH180"
   - Update this unit each turn as more items are completed.
   - Mark as "hard" priority so the agent doesn't forget remaining items.

4. LINK CASCADE / DERIVED dependencies: When a new unit's validity depends on another unit's status or value, surface that link.
   - Use `coupled_with: [unit_id]` when this unit's validity depends on the STATUS of another unit (active/closed/superseded).
     Example: "Netflix autopay on Card 3" is coupled_with the unit declaring Card 3's status -- if Card 3 closes, the Netflix routing must be re-evaluated.
   - Use `derived_from: [unit_id]` when this unit's VALUE was computed from another unit's value.
     Example: "monthly savings $1,800" is derived_from the unit "take-home pay $6,000" via a savings-rate rule -- if take-home changes, savings derives differently.
   - Use `triggers: [{"kind": "date_passed", "payload": {"date": "YYYY-MM-DD"}}]` when this unit becomes operationally relevant on a specific date or condition.
     Allowed kinds: "date_passed", "entity_closed", "supersession_announced", "cascade".

5. NO CHANGE: If the turn contains no new state-relevant information, or if the state already captures everything.

## Rules:
- Only create units for information RELEVANT to future decisions
- Do NOT duplicate information already in an existing unit -- check the current state first
- Priority "hard" for: verified facts, binding commitments, remaining obligations, things that constrain future actions
- Priority "soft" for: preferences, nice-to-haves, things that can yield
- Keep content concise -- one clear sentence per unit
- Type and scope are free-form labels you assign
- For coupled_with / derived_from / triggers: only include when there is a REAL dependency. Leave empty otherwise.

## Output format:
Return a JSON object:
{
  "add": [
    {
      "content": "string",
      "priority": "hard" | "soft",
      "source_type": "user" | "action" | "tool_result",
      "type": "string (free-form category label)",
      "scope": "string (what this applies to)",
      "coupled_with": ["unit_id_a", "unit_id_b"],
      "derived_from": ["unit_id_c"],
      "triggers": [{"kind": "date_passed", "payload": {"date": "2026-06-15"}, "description": "tax deadline"}]
    }
  ],
  "supersede": [
    {
      "unit_id": "string (ID of existing unit to supersede)",
      "reason": "string (why it's being superseded)",
      "replacement": {
        "content": "string (new version)",
        "priority": "hard" | "soft",
        "source_type": "user" | "action" | "tool_result",
        "type": "string",
        "scope": "string",
        "coupled_with": [],
        "derived_from": [],
        "triggers": []
      }
    }
  ]
}

If no updates are needed, return: {"add": [], "supersede": []}

Output ONLY the JSON object, no other text."""
# --- end of verbatim part ---


# ------------------------------------------------------------------ state store + deterministic rechecker
class Store:
    """StateMem's StateStore and Rechecker as we read sections 5.1-5.3 (choices the paper leaves open are listed in
    PREREG_MEME_WRITE.md B2). Units: id, content, priority, type, source, date, deps (ids), status, recheck (trigger)."""

    def __init__(self):
        self.units = []

    def get(self, uid):
        return next((u for u in self.units if u["id"] == uid), None)

    def active(self):
        return [u for u in self.units if u["status"] == "active"]

    def add(self, content, priority="hard", type="fact", source="", date="", coupled_with=(), derived_from=(), triggers=()):
        ok = lambda ids: [i for i in (ids if isinstance(ids, (list, tuple)) else []) if isinstance(i, str) and self.get(i) and self.get(i)["status"] == "active"]
        u = dict(id=f"u_{len(self.units) + 1:04d}", content=" ".join(str(content).split()), priority="soft" if priority == "soft" else "hard",
                 type=" ".join(str(type or "fact").split())[:40], source=source, date=date, coupled_with=ok(coupled_with),
                 derived_from=ok(derived_from), triggers=list(triggers) if isinstance(triggers, (list, tuple)) else [], status="active", recheck=None)
        self.units.append(u)
        return u["id"]

    def supersede(self, uid, reason="", replacement=None, source="", date=""):
        """mark `uid` superseded, add the replacement (if any), flag every active unit that depends on it (transitively)"""
        old = self.get(uid)
        if old is None or old["status"] != "active":
            return None
        old["status"] = "superseded"
        new = None
        if isinstance(replacement, dict) and str(replacement.get("content", "")).strip():
            new = self.add(replacement["content"], replacement.get("priority", old["priority"]), replacement.get("type", old["type"]), source, date,
                           replacement.get("coupled_with", ()), replacement.get("derived_from", ()), replacement.get("triggers", ()))
        why = f'"{old["content"]}" was superseded' + (f" ({' '.join(str(reason).split())})" if reason else "") + (f'; now: "{self.get(new)["content"]}"' if new else "")
        frontier = [(uid, why)]
        while frontier:                                          # breadth-first over the dependency edges
            cur, trig = frontier.pop(0)
            for u in self.active():
                if u["id"] != new and cur in u["coupled_with"] + u["derived_from"] and u["recheck"] is None:
                    u["recheck"] = trig
                    frontier.append((u["id"], f'depends on a unit that needs recheck: "{u["content"]}"'))
        if new:                                                  # dependents now point at the replacement
            for u in self.active():
                for k in ("coupled_with", "derived_from"):
                    u[k] = [new if i == uid else i for i in u[k]]
        return new

    def apply(self, upd, source="", date=""):
        """apply one TurnEncoder output {"add": [...], "supersede": [...]}; returns (n_added, n_superseded)"""
        na = ns = 0
        for a in upd.get("add") or []:
            if isinstance(a, dict) and str(a.get("content", "")).strip():
                self.add(a["content"], a.get("priority"), a.get("type"), source, date, a.get("coupled_with", ()), a.get("derived_from", ()), a.get("triggers", ()))
                na += 1
        for s in upd.get("supersede") or []:
            u = self.get(str(s.get("unit_id", ""))) if isinstance(s, dict) else None
            if u and u["status"] == "active":
                self.supersede(u["id"], s.get("reason", ""), s.get("replacement"), source, date)
                ns += 1
        return na, ns

    def compact(self):
        """what the TurnEncoder sees as the current state: one line per active unit, with ids"""
        out = []
        for u in self.active():
            dep = "; ".join(f"{k}: {', '.join(u[k])}" for k in ("coupled_with", "derived_from") if u[k])
            out.append(f'- [{u["id"]}] ({u["priority"]}) [{u["type"]}] {u["content"]}' + (f" {{{dep}}}" if dep else "") + (" [NEEDS RECHECK]" if u["recheck"] else ""))
        return "\n".join(out) or "(empty)"

    def block(self):
        """Appendix E.3 block layout (deterministic renderer); empty sections are left out"""
        line = lambda u: f'- [{u["type"]}] {u["content"]}' + (f' ({u["date"]})' if u["date"] else "")
        act = self.active()
        secs = [("## BINDING -- THIS CONVERSATION", [line(u) for u in act if u["priority"] == "hard" and not u["recheck"]]),
                ("## PREFERENCES (yield if necessary)", [line(u) for u in act if u["priority"] == "soft" and not u["recheck"]]),
                ("## ! NEEDS RECHECK -- may be stale due to recent events", [line(u) + f'\n  -> trigger: {u["recheck"]}' for u in act if u["recheck"]])]
        return "\n\n".join(h + "\n" + "\n".join(ls) for h, ls in secs if ls) or "(no state recorded)"


def parse_update(text):
    """the JSON object of a TurnEncoder reply, or None"""
    t = re.sub(r"<think>.*?</think>", "", text, flags=re.S)
    a, b = t.find("{"), t.rfind("}")
    if a < 0 or b <= a:
        return None
    try:
        d = json.loads(t[a:b + 1])
    except json.JSONDecodeError:
        return None
    return d if isinstance(d, dict) else None


# ------------------------------------------------------------------ step 1: oracle write products
def name(e):
    return e.replace("_", " ")


def facts(ep, pos):
    """annotated facts of the evidence sessions up to session index `pos`, in order"""
    return [g | dict(si=si, date=s["timestamp"].split(" ")[0]) for si, s in enumerate(ep["sessions"]) if si <= pos and s["type"] != "filler"
            for g in s["gold_facts"]]


def pick(F, e, kind):
    for g in F:
        if g["entity"] == e and ((kind == "dep" and g.get("has_dependency") and not g.get("is_if_then")) or (kind == "rule" and g.get("is_if_then"))
                                 or (kind == "delete" and g.get("type") == "delete")):
            return g["fact_text"]
    return ""


def chain(ep, e):
    """[(entity, its upstream)] from the root's child down to the target: one pair for 1-hop, two for 2-hop"""
    src = ep["entities"][e].get("cascade_source") or ep["root"]
    return [(e, src)] if src == ep["root"] else [(src, ep["root"]), (e, src)]


def golden_after(ep, q, F):
    """w4: MEME golden_memory.py context builders (verbatim line templates)"""
    e, tt, root, rc = q["entity"][0], q["task_type"], ep["root"], ep["root_change"]
    if tt == "Del":
        d = pick(F, e, "delete")
        return "Related information:\n" + "\n".join([f"- User previously mentioned their {name(e)}: '{q['entity_values'][e]}'.",
                                                     f'- User later requested: "{d}"' if d else "- User later requested to delete/forget this information."])
    lines = [f"- User's {name(root)} changed from '{rc['before']}' to '{rc['after']}'."]
    if tt == "Cas":
        lines.append(f"- User's {name(e)} was previously '{ep['entities'][e]['before']}'.")
        if pick(F, e, "dep"):
            lines.append(f'- Dependency: "{pick(F, e, "dep")}"')
        if pick(F, e, "rule"):
            lines.append(f'- Rule: "{pick(F, e, "rule")}"')
        lines.append(f"- User's current {name(e)}: {q['gold_answer']}")
    else:
        if pick(F, e, "dep"):
            lines.append(f'- Previously stated: "{pick(F, e, "dep")}"')
        lines.append(f"- User's {name(e)} was previously '{q['entity_values'][e]}', but no new value was declared after the change.")
        lines.append(f"- Since {name(root)} changed and {name(e)} depended on it, the previous value may no longer be valid.")
    return "Related information:\n" + "\n".join(lines)


def entity_lines(ep, x, F, before):
    out = [f"- User's current {name(x)}: {ep['entities'][x]['before']}" if before else f"- User's {name(x)} was previously '{ep['entities'][x]['before']}'."]
    if pick(F, x, "dep"):
        out.append(f'- Dependency: "{pick(F, x, "dep")}"')
    if pick(F, x, "rule"):
        out.append(f'- Rule: "{pick(F, x, "rule")}"')
    return out


def flag_after(ep, q, F):
    """w3: the same lines as w4 without the computed value; one task-agnostic flag per entity on the chain"""
    e, root, rc = q["entity"][0], ep["root"], ep["root_change"]
    lines = [f"- User's {name(root)} changed from '{rc['before']}' to '{rc['after']}'."]
    for i, (x, src) in enumerate(chain(ep, e)):
        lines += entity_lines(ep, x, F, False)
        lines.append(f"- NEEDS RECHECK: this value depends on {name(src)}, which " + ("changed after the value was stated." if i == 0 else "is flagged NEEDS RECHECK above."))
    return "Related information:\n" + "\n".join(lines)


def state_before(ep, q, F):
    """the pre-change state in the same line format (shared by w3 and w4)"""
    e = q["entity"][0]
    if q["task_type"] == "Del":
        return f"Related information:\n- User's current {name(e)}: {q['entity_values'][e]}"
    lines = [f"- User's current {name(ep['root'])}: {ep['root_change']['before']}"]
    for x, _ in chain(ep, e):
        lines += entity_lines(ep, x, F, True)
    return "Related information:\n" + "\n".join(lines)


def oracle_store(ep, pos):
    """w3s: every annotated fact up to `pos` as a StateMem unit, written by a perfect writer"""
    st, cur, val = Store(), {}, {}
    for g in facts(ep, pos):
        e, text, kw = g["entity"], g["fact_text"], dict(source=f"session {g['si'] + 1}", date=g["date"])
        if g.get("type") == "delete" and e in cur:
            cur[e] = st.supersede(cur[e], "the user asked to delete it", dict(content=f"The user asked to delete the {name(e)}; it has been removed from memory.", type="deletion"), **kw)
        elif g.get("is_if_then"):
            st.add(text, type="rule", **kw)                      # a standing rule is not stale when its condition fires
        elif e not in cur:
            src = cur.get(g.get("dependency_source"))
            cur[e], val[e] = st.add(text, type="fact", coupled_with=[src] if src else [], **kw), g["value"]
        elif g["value"] == val[e]:                               # the same value restated with its dependency: refine in place
            u, src = st.get(cur[e]), cur.get(g.get("dependency_source"))
            u["content"] = " ".join(text.split())
            if src and src not in u["coupled_with"]:
                u["coupled_with"].append(src)
        else:                                                    # a new value: supersede, the rechecker flags the dependents
            cur[e], val[e] = st.supersede(cur[e], "the user stated a new value", dict(content=text, type="fact"), **kw), g["value"]
    return st


def products(ep, rag=None):
    """{(phase, cond, entity): (question, context)}; `rag` = meme_reader.contexts(ep) to add the combined conditions"""
    out = {}
    bpos, apos = ep["before_questions"]["position_after_session"], ep["after_questions"]["position_after_session"]
    Fb, Fa = facts(ep, bpos), facts(ep, apos)
    sb, sa = oracle_store(ep, bpos).block() + "\n\n" + RECOMPUTE_GUIDANCE, oracle_store(ep, apos).block() + "\n\n" + RECOMPUTE_GUIDANCE
    before = {q["entity"][0]: q for q in ep["before_questions"]["questions"] if q["task_type"] in DEP_TASKS}
    for q in M.questions(ep):
        e, b = q["entity"][0], before[q["entity"][0]]
        out["before", "w4", e] = (b, state_before(ep, b, Fb))
        out["after", "w4", e] = (q, golden_after(ep, q, Fa))
        if q["task_type"] != "Del":
            out["after", "w3", e] = (q, flag_after(ep, q, Fa))
        out["before", "w3s", e], out["after", "w3s", e] = (b, sb), (q, sa)
        if rag is not None:
            out["before", "w4rag", e] = (b, out["before", "w4", e][1] + CONNECT + rag["before", "rag", e][1])
            out["after", "w4rag", e] = (q, out["after", "w4", e][1] + CONNECT + rag["after", "rag", e][1])
            if q["task_type"] != "Del":
                out["after", "w3rag", e] = (q, out["after", "w3", e][1] + CONNECT + rag["after", "rag", e][1])
    return out


def mislabeled(eps):
    """Absence items whose target has an if-then rule in the conversation (MEME's known label issue): {(ep, entity)}"""
    return {(ep["episode_id"], q["entity"][0]) for ep in eps for q in M.questions(ep)
            if q["task_type"] == "Abs" and pick(facts(ep, ep["after_questions"]["position_after_session"]), q["entity"][0], "rule")}


# ------------------------------------------------------------------ running the reader
def run_rows(tag, think, jobs, model_tok=None):
    """jobs: iterable of (row dict, context); same prompt, decoding and row schema as meme_reader.answer"""
    import torch
    from app_common import JsonlAppender, load_reader
    from explore_probe import LONG_SDPA, sdpa_kernel
    from run_app_fix import generate, generate_think
    w = JsonlAppender(R.ANS.format(tag + ("_think" if think else "_off")), key=R.key)
    tok, model = model_tok or load_reader(R.READER)
    for row, q, c in jobs:
        if R.key(row) in w.done:
            continue
        p = R.UNIFIED_ANSWER_PROMPT.format(context=c, question=q["question"])
        chat = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=think)
        with sdpa_kernel(LONG_SDPA, set_priority=True), torch.no_grad():
            full = generate_think(tok, model, chat, R.THINK_EXTRA + R.GEN, ("meme-reader", R.key(row))) if think else generate(tok, model, chat, R.GEN)
        fin = ("</think>" in full) if think else True
        hyp = (full.split("</think>", 1)[1] if fin else "").strip() if think else full.strip()
        e = row["entity"]
        w.write(row | dict(question=q["question"], gold=q["entity_values"][e], n_tok=len(tok(chat).input_ids), finished=fin, answer=hyp,
                           n_out=len(tok(full, add_special_tokens=False).input_ids)))
        torch.cuda.empty_cache()
    w.close()


def jobs_for(eps, conds, build):
    for ep in eps:
        for (phase, cond, e), (q, c) in build(ep).items():
            if cond in conds:
                yield dict(ep=ep["episode_id"], phase=phase, cond=cond, entity=e, task=q["task_type"], hop=q.get("hop", 0)), q, c


def arg(name, default=None):
    v = [a.split("=", 1)[1] for a in sys.argv if a.startswith(f"--{name}=")]
    return v[0] if v else default


def answer():
    conds = arg("conds").split(",")
    need_rag = any(c.endswith("rag") for c in conds)
    run_rows("w", "--think" in sys.argv, jobs_for(M.load(), conds, lambda ep: products(ep, R.contexts(ep) if need_rag else None)))


def dry():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M.EMB)
    eps, n, miss = M.load(), {}, {}
    for ep in eps:
        Fa = facts(ep, ep["after_questions"]["position_after_session"])
        for q in M.questions(ep):
            e = q["entity"][0]
            for kind in (("delete",) if q["task_type"] == "Del" else ("dep", "rule")):
                miss.setdefault((q["task_type"], kind, bool(pick(Fa, e, kind))), []).append(1)
            if q["task_type"] != "Del":
                assert all(x in ep["entities"] for x, _ in chain(ep, e)) and len(chain(ep, e)) == q["hop"], (ep["episode_id"], e)
        for (phase, cond, e), (q, c) in products(ep, R.contexts(ep)).items():
            n.setdefault((phase, cond, q["task_type"]), []).append(len(tok(c, add_special_tokens=False).input_ids))
            if phase == "after" and cond in ("w4", "w3") and q["task_type"] == "Cas":     # the computed value is in w4 and only in the rule in w3
                assert q["gold_answer"] in c and (cond == "w4") == (f"current {name(e)}: {q['gold_answer']}" in c)
        sa = oracle_store(ep, ep["after_questions"]["position_after_session"])
        fl = sum(1 for u in sa.active() if u["recheck"])
        n.setdefault(("store", "active units", ""), []).append(len(sa.active()))
        n.setdefault(("store", "flagged units", ""), []).append(fl)
        tg = [q for q in M.questions(ep) if q["task_type"] != "Del"]
        n.setdefault(("store", "targets flagged %", ""), []).append(100 * np.mean([any(u["recheck"] and ep["entities"][q["entity"][0]]["before"] in u["content"]
                                                                                 for u in sa.active()) for q in tg]))
    for k, v in sorted(n.items()):
        print(k, "n", len(v), "median", int(np.median(v)), "min", int(min(v)), "max", int(max(v)))
    print("line available:", {k: len(v) for k, v in sorted(miss.items())})
    print("Absence items whose target has an if-then rule (known label issue):", sorted(mislabeled(eps)))


# ------------------------------------------------------------------ step 2a: StateMem with a local writer
def sm_paths(writer):
    t = "sm" + re.sub(r"\D", "", writer.split("-")[-1])           # Qwen3-8B -> sm8
    return t, os.path.join(M.OUT, f"{t}_log.jsonl"), os.path.join(M.OUT, t + "_{}.json")


def turns(ep):
    """messages visible at the after-phase, in order: (index, session index, date, role, content)"""
    last, out = ep["after_questions"]["position_after_session"], []
    for si, s in enumerate(ep["sessions"]):
        if si > last:
            break
        for t in s["conversation"]:
            out.append((len(out), si, s["timestamp"].split(" ")[0], t["role"], t["content"]))
    return out


def ingest():
    import torch
    from app_common import JsonlAppender, load_jsonl, load_reader
    from explore_probe import LONG_SDPA, sdpa_kernel
    from run_app_fix import generate
    writer = arg("writer", "Qwen3-8B")
    tag, log_path, snap = sm_paths(writer)
    want = (arg("eps") or ",".join(PILOT)).split(",")
    w = JsonlAppender(log_path, key=lambda r: f"{r['ep']}|{r['i']}")
    logged = {(r["ep"], r["i"]): r for r in load_jsonl(log_path)}
    tok = model = None
    for ep in [e for e in M.load() if e["episode_id"] in want]:
        eid = ep["episode_id"]
        if os.path.exists(snap.format(eid)):
            continue
        st, shots, bpos = Store(), {}, ep["before_questions"]["position_after_session"]
        T = turns(ep)
        for i, si, date, role, content in T:
            if "before" not in shots and si > bpos:
                shots["before"] = dict(block=st.block(), units=json.loads(json.dumps(st.units)))
            r = logged.get((eid, i))
            if r is None:
                if model is None:
                    tok, model = load_reader(writer)
                p = ENC_PROMPT.replace("<<current_state>>", st.compact()).replace("<<turn_number>>", str(i + 1)).replace("<<latest_turn>>", f"{role}: {content}")
                chat = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=False)
                t0 = time.time()
                with sdpa_kernel(LONG_SDPA, set_priority=True), torch.no_grad():
                    raw = generate(tok, model, chat, ENC_GEN)
                r = dict(ep=eid, i=i, si=si, role=role, raw=raw, n_tok=len(tok(chat).input_ids), n_out=len(tok(raw, add_special_tokens=False).input_ids),
                         secs=round(time.time() - t0, 2), n_active=len(st.active()))
                w.write(r)
                torch.cuda.empty_cache()
            upd = parse_update(r["raw"])
            if upd is not None:
                st.apply(upd, source=f"turn {i + 1} ({role})", date=date)
        shots["after"] = dict(block=st.block(), units=st.units)
        json.dump(shots, open(snap.format(eid), "w", encoding="utf-8"), ensure_ascii=False)
        print(eid, "turns", len(T), "active units", len(st.active()), "flagged", sum(1 for u in st.active() if u["recheck"]), flush=True)
    w.close()


def sm_contexts(ep, snap):
    """smu: the state block as retrieved context (MEME protocol); smg: block + StateMem's recompute guidance"""
    s, out = json.load(open(snap.format(ep["episode_id"]), encoding="utf-8")), {}
    for phase in ("before", "after"):
        for q in ep[phase + "_questions"]["questions"]:
            if q["task_type"] in DEP_TASKS + ("ER",):
                out[phase, "smu", q["entity"][0]] = (q, s[phase]["block"])
                out[phase, "smg", q["entity"][0]] = (q, s[phase]["block"] + "\n\n" + RECOMPUTE_GUIDANCE)
    return out


def smread():
    tag, _, snap = sm_paths(arg("writer", "Qwen3-8B"))
    eps = [e for e in M.load() if os.path.exists(snap.format(e["episode_id"]))]
    run_rows(tag, "--think" in sys.argv, jobs_for(eps, ("smu", "smg"), lambda ep: sm_contexts(ep, snap)))


def check():
    """ingestion statistics and the string-based write check on the after-phase snapshot (aggregates only)"""
    from app_common import load_jsonl
    tag, log_path, snap = sm_paths(arg("writer", "Qwen3-8B"))
    log = load_jsonl(log_path)
    if not log:
        return print("no ingestion log")
    ok = [parse_update(r["raw"]) is not None for r in log]
    secs, ntok = np.array([r["secs"] for r in log]), np.array([r["n_tok"] for r in log])
    print(f"{tag}: calls {len(log)}, JSON parse failures {100 - np.mean(ok) * 100:.1f}%, hit the {ENC_GEN}-token cap {np.mean([r['n_out'] >= ENC_GEN for r in log]) * 100:.1f}%")
    print(f"  seconds per call median {np.median(secs):.1f} mean {secs.mean():.1f} total {secs.sum() / 3600:.2f} h | prompt tokens median {int(np.median(ntok))} max {int(ntok.max())}")
    done = [e for e in M.load() if os.path.exists(snap.format(e["episode_id"]))]
    per_turn = secs.sum() / len(log)
    print(f"  projected for all 30 episodes (6341 messages): {per_turn * 6341 / 3600:.1f} h")
    low, rows, roots = (lambda s: " ".join(str(s).lower().split())), [], []
    hypo = re.compile(r"\b(if|will|would)\b")
    for ep in done:
        act = [u for u in json.load(open(snap.format(ep["episode_id"]), encoding="utf-8"))["after"]["units"] if u["status"] == "active"]
        has = lambda v, us: any(low(v) in low(u["content"]) for u in us)
        roots.append(has(ep["root_change"]["after"], act))
        print(f"  {ep['episode_id']}: active units {len(act)}, flagged {sum(1 for u in act if u['recheck'])}, with a dependency edge {sum(1 for u in act if u['coupled_with'] or u['derived_from'])}")
        for q in M.questions(ep):
            e = q["entity"][0]
            old = q["entity_values"][e] if q["task_type"] != "Cas" else ep["entities"][e]["before"]
            rows.append(dict(task=q["task_type"], old_flagged=has(old, [u for u in act if u["recheck"]]), old_unflagged=has(old, [u for u in act if not u["recheck"]]),
                             new_computed=q["task_type"] == "Cas" and has(q["gold_answer"], [u for u in act if not hypo.search(low(u["content"]))]),
                             new_anywhere=q["task_type"] == "Cas" and has(q["gold_answer"], act)))
    import pandas as pd
    df = pd.DataFrame(rows)
    print(f"  root's new value in an active unit: {sum(roots)} / {len(roots)} episodes")
    print("  write check per question (counts): n, old value in a flagged unit, old value in an unflagged active unit, Cas new value in a non-hypothetical unit, Cas new value anywhere")
    print(df.groupby("task").agg(n=("task", "size"), old_flagged=("old_flagged", "sum"), old_unflagged=("old_unflagged", "sum"),
                                 new_computed=("new_computed", "sum"), new_anywhere=("new_anywhere", "sum")).to_string())


# ------------------------------------------------------------------ tables
def table():
    import glob
    import pandas as pd
    from app_common import load_jsonl
    rng, bad, per_q, rows = np.random.default_rng(0), mislabeled(M.load()), {}, []

    def boot(t, col):
        if not len(t):
            return "n=0"
        eps = sorted(t.ep.unique())
        s = t[col].astype(float).groupby(t.ep).agg(["sum", "count"]).reindex(eps).values
        idx = rng.integers(0, len(eps), (10000, len(eps)))
        v = s[idx, 0].sum(1) / np.maximum(s[idx, 1].sum(1), 1) * 100
        return f"{t[col].mean() * 100:+6.1f} [{np.percentile(v, 2.5):.0f}, {np.percentile(v, 97.5):.0f}]"

    for path in sorted(glob.glob(R.ANS.format("*_judged"))):
        tag = os.path.basename(path)[len("reader_"):-len("_judged.jsonl")]
        df = pd.DataFrame(load_jsonl(path))
        er, df = df[df.task == "ER"], df[df.task != "ER"]
        bef = df[df.phase == "before"].set_index(["ep", "entity", "cond"]).correct
        aft = df[df.phase == "after"].copy()
        aft["knew"] = [bool(bef.get((e, n, BEFORE_OF.get(c, "rag" if c == "hop" else c)), False)) for e, n, c in zip(aft.ep, aft.entity, aft.cond)]
        aft["real"] = aft.correct & aft.knew
        for cond in sorted(aft.cond.unique()):
            g = aft[aft.cond == cond]
            per_q[tag, cond] = g.set_index(["ep", "entity"])[["task", "real"]]
            if not (re.match(r"(w|sm\d+)_", tag) or "--all" in sys.argv):
                continue
            row = dict(tag=tag, cond=cond, eps=g.ep.nunique())
            for task in DEP_TASKS:
                t = g[g.task == task]
                row[task], row[task + "_raw"], row[task + "_knew"] = boot(t, "real"), round(t.correct.mean() * 100, 1), round(t.knew.mean() * 100, 1)
            c1, clean = g[(g.task == "Cas") & (g.hop == 1)], g[(g.task == "Abs") & np.array([k not in bad for k in zip(g.ep, g.entity)], dtype=bool)]
            row["Cas_1hop"], row["Cas_2hop"] = round(c1.real.mean() * 100, 1), round(g[(g.task == "Cas") & (g.hop == 2)].real.mean() * 100, 1)
            row["Abs_1hop"], row["Abs_clean"] = round(g[(g.task == "Abs") & (g.hop == 1)].real.mean() * 100, 1), round(clean.real.mean() * 100, 1)
            row["unfinished"] = int((~df[df.cond == cond].finished).sum())
            e = er[er.cond == cond]
            row["ER"] = f"{e.correct.mean() * 100:.1f} (n={len(e)})" if len(e) else ""
            rows.append(row)
    pd.set_option("display.width", 320)
    pd.set_option("display.max_columns", 40)
    print(pd.DataFrame(rows).to_string(index=False) if rows else "no write-side results yet")
    print("\npaired differences in real pass, a minus b, over the questions both have (episode bootstrap):")
    for pr in (arg("pairs") or "").split(","):
        if ":" not in pr:
            continue
        (ta, ca), (tb, cb) = [x.split("/") for x in pr.split(":")]
        if (ta, ca) not in per_q or (tb, cb) not in per_q:
            print(f"  {pr}: missing")
            continue
        x, y = per_q[ta, ca], per_q[tb, cb]
        idx = x.index.intersection(y.index)
        d = pd.DataFrame(dict(ep=[i[0] for i in idx], task=x.loc[idx].task.values, d=x.loc[idx].real.astype(float).values - y.loc[idx].real.astype(float).values))
        print(f"  {pr}: " + " | ".join(f"{task} {boot(d[d.task == task], 'd')} (n={int((d.task == task).sum())})" for task in DEP_TASKS))


def selftest():
    """no GPU: the store, the rechecker, the parser and both renderers on made-up units"""
    st = Store()
    a = st.add("take-home pay is $6000", type="income")
    b = st.add("savings rate is 30%", type="rule")
    c = st.add("monthly savings $1,800", type="derived", derived_from=[a, b, "u_9999"])
    d = st.add("vacation budget is what is left after savings", priority="soft", type="budget", coupled_with=[c])
    assert st.get(c)["derived_from"] == [a, b]
    upd = parse_update('```json\n{"add": [], "supersede": [{"unit_id": "%s", "reason": "raise", "replacement": {"content": "take-home pay is $7500", "priority": "hard", "type": "income"}}]}\n```' % a)
    assert st.apply(upd) == (0, 1)
    assert st.get(a)["status"] == "superseded" and st.get(c)["recheck"] and st.get(d)["recheck"] and not st.get(b)["recheck"]
    assert st.get(c)["derived_from"] == ["u_0005", b] and parse_update("no json here") is None and st.supersede(a) is None
    assert st.apply({"add": [{"content": "x"}, "junk", {"content": ""}], "supersede": [{"unit_id": "nope"}, 3]}) == (1, 0)
    print(st.compact(), "\n")
    print(st.block())
    print("\nselftest ok")


if __name__ == "__main__":
    s = sys.argv[1]
    if s in ("answer", "ingest", "smread") and os.name == "nt":     # process-scoped keep-awake, released on exit; no system setting changes
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    {"dry": dry, "selftest": selftest, "answer": answer, "ingest": ingest, "smread": smread, "check": check, "table": table}[s]()
