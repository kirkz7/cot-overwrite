"""MEME reader pilot (diagnosis only): with the retrieval of meme_pilot.py held fixed, how often does a local reader
answer Cascade / Absence / Deletion questions correctly? Design and thresholds: PREREG_MEME_READER.md (written first).
Exploration episodes only. Answer prompt and judge prompts are MEME's own (code/agents/base.py, code/eval/judge.py);
the judge model is local Qwen3-14B instead of gpt-4o. No conversation text is printed, aggregates only.
usage: python 论文/meme_reader.py dry | answer [--think] | judge | stats
"""
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
import meme_pilot as M

READER, JUDGE, GEN, THINK_EXTRA = "Qwen3-8B", "Qwen3-14B", 500, 1024
ANS = os.path.join(M.OUT, "reader_{}.jsonl")          # tag: off | think
CONDS = {"after": ("rag", "hop", "oracle"), "before": ("rag", "oracle")}

# --- verbatim from MEME-public (MIT): code/agents/base.py and code/eval/judge.py ---
UNIFIED_ANSWER_PROMPT = """Answer the user's question based ONLY on the context provided below.
If the information is not in the context, say you don't have that information.
Answer with ONLY the value. Do not explain or add context.

Context:
{context}

Question: {question}"""

BEFORE_COMMON_PROMPT = """You are a judge evaluating whether an AI agent's answer is semantically correct.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

Does the agent's answer contain the correct information matching the gold answer?

Rules:
- Focus on semantic equivalence, not exact wording
- If the gold value is present in the agent's answer, it is correct — regardless of any additional information, future possibilities, or extra details the agent mentions
- "Dentist every 6 months; dermatologist monthly (if you change residence)" → gold is "dentist (every 6 months)" → YES (core answer correct, extra info is irrelevant)
- "40 minutes" = "40 min" → YES
- "dentist appointment every 6 months" = "dentist (every 6 months)" → YES
- If agent says "I don't know" and gold is a specific value → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

CASCADE_D_PROMPT = """You are a judge evaluating whether an AI agent correctly reports a CASCADE-DEPENDENT value after an upstream change.

QUESTION: {question}
GOLD (new cascaded value): {gold}
AGENT: {agent_answer}

The upstream entity changed, triggering a cascade. The agent should now report ONLY the new value.

Rules:
- Agent states the new value as the sole, definitive answer → YES
- Agent lists BOTH old and new values together → NO
- Agent states only the old value → NO
- Agent lists multiple options or hedges ("might", "considering") → NO
- Agent uses future tense ("will change to X", "will be X") instead of confirming as current → NO
- "I don't know" → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

CASCADE_U_PROMPT = """You are a judge evaluating whether an AI agent appropriately handles UNCERTAINTY after an upstream entity changed.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

The upstream entity changed, making the previous answer potentially invalid. The agent should express uncertainty.

Rules:
- Agent says "I don't know" / "not sure" / "Unknown" / "None" / expresses uncertainty → YES
- Agent acknowledges the upstream change and questions validity → YES
- Agent confidently states the OLD value as definitive current fact → NO
- Agent provides the old value with no hedging or uncertainty → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""

DELETION_PROMPT = """You are a judge evaluating whether an AI agent correctly handles a DELETED piece of information.

QUESTION: {question}
GOLD: {gold}
AGENT: {agent_answer}

The information was explicitly deleted/removed. The agent should not return it.

Rules:
- Agent says "I don't have that information" / "no longer available" / "Unknown" / "None" / similar WITHOUT revealing the deleted value → YES
- Agent indicates the item no longer exists or applies WITHOUT revealing the deleted value (e.g., "No food allergies" when the deleted value was a specific allergy) → YES
- Agent reveals the deleted value in any way (even while saying it was deleted/removed) → NO
- Agent returns the deleted value as if it still exists → NO

Output ONLY JSON: {{"correct": true/false, "reason": "one sentence"}}"""
JUDGE_PROMPTS = {"Cas": CASCADE_D_PROMPT, "Abs": CASCADE_U_PROMPT, "Del": DELETION_PROMPT}
# --- end of verbatim part ---


def render(ep, spans, keep):
    """the kept messages (indices into spans) as whole messages, chronological, under their session header"""
    out, cur = [], None
    for i in sorted(keep):
        si, ti, role, _, _ = spans[i]
        if si != cur:
            out.append(("\n" if out else "") + f"[Session {si + 1} | {ep['sessions'][si]['timestamp']}]")
            cur = si
        out.append(f"{role}: {ep['sessions'][si]['conversation'][ti]['content']}")
    return "\n".join(out)


def contexts(ep):
    """{(phase, cond, entity): (question dict, context text)} for one episode"""
    text, spans = M.serialize(ep)
    d = json.load(open(os.path.join(M.OUT, f"dense_{ep['episode_id']}.json")))
    r = json.load(open(os.path.join(M.OUT, f"rerank_{ep['episode_id']}.json")))
    wins = d["windows"]
    msgs = lambda sel, last: {i for i, (si, _, _, a, b) in enumerate(spans) if si <= last and any(wins[w][0] < b and a < wins[w][1] for w in sel)}
    evid = lambda last: {i for i, (si, *_) in enumerate(spans) if si <= last and ep["sessions"][si]["type"] == "evidence"}
    out = {}
    bpos, apos = ep["before_questions"]["position_after_session"], ep["after_questions"]["position_after_session"]
    bend = max(b for si, _, _, _, b in spans if si <= bpos)
    before = {q["entity"][0]: q for q in ep["before_questions"]["questions"] if q["task_type"] in ("Cas", "Abs", "Del")}
    for q in M.questions(ep):
        e, order = q["entity"][0], r[q["question"]]
        sel = order[:M.BASE_K]
        anc = M.ancestors(ep, q)
        for a in anc:                                            # same 9 + 6 rule as meme_pilot.stats
            sel = sel + [w for w in r[M.hop_query(ep, a)] if w not in sel][:(M.FINAL_K - M.BASE_K) // len(anc)]
        out["after", "rag", e] = (q, render(ep, spans, msgs(order[:M.FINAL_K], apos)))
        out["after", "hop", e] = (q, render(ep, spans, msgs(sel if anc else order[:M.FINAL_K], apos)))
        out["after", "oracle", e] = (q, render(ep, spans, evid(apos)))
        b = before[e]
        assert b["question"] == q["question"]
        out["before", "rag", e] = (b, render(ep, spans, msgs([w for w in order if wins[w][0] < bend][:M.FINAL_K], bpos)))
        out["before", "oracle", e] = (b, render(ep, spans, evid(bpos)))
    return out


def key(r):
    return f"{r['ep']}|{r['phase']}|{r['cond']}|{r['entity']}"


def dry():
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(M.EMB)
    n = {}
    for ep in M.load():
        for (phase, cond, e), (q, c) in contexts(ep).items():
            n.setdefault((phase, cond), []).append(len(tok(c, add_special_tokens=False).input_ids))
    for k, v in n.items():
        print(k, "n", len(v), "tokens median", int(np.median(v)), "p90", int(np.percentile(v, 90)), "max", max(v))


def answer(think):
    import torch
    from app_common import JsonlAppender, load_reader
    from explore_probe import LONG_SDPA, sdpa_kernel
    from run_app_fix import generate, generate_think
    conds = [a.split("=")[1].split(",") for a in sys.argv if a.startswith("--conds=")]     # run order only; default all
    w = JsonlAppender(ANS.format("think" if think else "off"), key=key)
    tok, model = load_reader(READER)
    for ep in M.load():
        for (phase, cond, e), (q, c) in contexts(ep).items():
            if conds and cond not in conds[0]:
                continue
            row = dict(ep=ep["episode_id"], phase=phase, cond=cond, entity=e, task=q["task_type"], hop=q.get("hop", 0))
            if key(row) in w.done:
                continue
            p = UNIFIED_ANSWER_PROMPT.format(context=c, question=q["question"])
            chat = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True, enable_thinking=think)
            with sdpa_kernel(LONG_SDPA, set_priority=True), torch.no_grad():
                full = generate_think(tok, model, chat, THINK_EXTRA + GEN, ("meme-reader", key(row))) if think else generate(tok, model, chat, GEN)
            fin = ("</think>" in full) if think else True
            hyp = (full.split("</think>", 1)[1] if fin else "").strip() if think else full.strip()
            w.write(row | dict(question=q["question"], gold=q["entity_values"][e], n_tok=len(tok(chat).input_ids), finished=fin,
                               answer=hyp, n_out=len(tok(full, add_special_tokens=False).input_ids)))
            torch.cuda.empty_cache()
        print(ep["episode_id"], "done", flush=True)
    w.close()


def judge():
    import torch
    from app_common import JsonlAppender, chat_prompt, load_jsonl, load_reader
    from probe import greedy
    tok = model = None
    for tag in ("off", "think"):
        path = ANS.format(tag)
        if not os.path.exists(path):
            continue
        w = JsonlAppender(path.replace(".jsonl", "_judged.jsonl"), key=key)
        todo = [r for r in load_jsonl(path) if key(r) not in w.done]
        print(tag, "todo", len(todo), flush=True)
        if todo and model is None:
            tok, model = load_reader(JUDGE)
            eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        for r in todo:
            tmpl = BEFORE_COMMON_PROMPT if r["phase"] == "before" else JUDGE_PROMPTS[r["task"]]
            p = tmpl.format(question=r["question"], gold=r["gold"], agent_answer=r["answer"])
            ids = torch.tensor([tok(chat_prompt(tok, p), add_special_tokens=False).input_ids], device="cuda")
            with torch.no_grad():
                reply = tok.decode(greedy(model, ids, 100, eos), skip_special_tokens=True)
            m = re.search(r'"correct"\s*:\s*(true|false)', reply.lower())
            w.write(r | dict(judge=reply.strip()[:300], parsed=bool(m), correct=bool(m and m.group(1) == "true")))
        w.close()


def norm(s):
    return " ".join(s.lower().split())


def stats():
    import pandas as pd
    from app_common import load_jsonl
    cov = pd.DataFrame(json.load(open(os.path.join(M.OUT, "per_question.json"))))[["ep", "entity", "located", "base_full", "hop_full"]]
    rng = np.random.default_rng(0)
    for tag in ("off", "think"):
        path = ANS.format(tag).replace(".jsonl", "_judged.jsonl")
        if not os.path.exists(path):
            continue
        df = pd.DataFrame(load_jsonl(path))
        print(f"\n######## reader {READER}, thinking {tag}: rows {len(df)}, judge output unparsed {int((~df.parsed).sum())}, "
              f"unfinished thinking {int((~df.finished).sum())}, prompt tokens median {int(df.n_tok.median())} max {int(df.n_tok.max())}, "
              f"output tokens median {int(df.n_out.median())}")
        # string cross-check of the judge (crisp cases only): before = gold value present; Cas = new value present;
        # Del = deleted value absent
        a, g = df.answer.map(norm), df.gold.map(norm)
        has = pd.Series([y in x for x, y in zip(a, g)], index=df.index)
        for name, m, rule in [("before", df.phase == "before", has), ("after Cas", (df.phase == "after") & (df.task == "Cas"), has),
                              ("after Del", (df.phase == "after") & (df.task == "Del"), ~has)]:
            print(f"  judge vs string rule, {name}: agree {((df.correct == rule)[m]).mean() * 100:.1f}% (n={int(m.sum())}; "
                  f"judge yes {df.correct[m].mean() * 100:.1f}%, rule yes {rule[m].mean() * 100:.1f}%)")
        bef = df[df.phase == "before"].set_index(["ep", "entity", "cond"]).correct
        aft = df[df.phase == "after"].merge(cov, on=["ep", "entity"], how="left")
        aft["knew"] = [bool(bef.get((e, n, "rag" if c == "hop" else c), False)) for e, n, c in zip(aft.ep, aft.entity, aft.cond)]
        aft["real"] = aft.correct & aft.knew
        eps = sorted(aft.ep.unique())

        def rate(g, col="real", g2=None):
            """percent and episode-bootstrap CI of mean(col) in g (minus the mean in g2 when given)"""
            def boot(x):
                s = x[col].astype(float).groupby(x.ep).agg(["sum", "count"]).reindex(eps, fill_value=0).values
                idx = rng_idx
                return s[idx, 0].sum(1) / np.maximum(s[idx, 1].sum(1), 1) * 100
            rng_idx = rng.integers(0, len(eps), (10000, len(eps)))
            if not len(g):
                return "n=0"
            v, m = boot(g), g[col].mean() * 100
            if g2 is not None:
                v, m = v - boot(g2), m - g2[col].mean() * 100
            return f"{m:5.1f} [{np.percentile(v, 2.5):.0f}, {np.percentile(v, 97.5):.0f}] (n={len(g)})"

        for task in ("Cas", "Abs", "Del"):
            t = aft[aft.task == task]
            print(f"\n  == {task}")
            for cond in CONDS["after"]:
                g = t[t.cond == cond]
                if task == "Del" and cond == "hop":
                    continue
                print(f"    {cond:6s} real pass {rate(g)} | raw pass {rate(g, 'correct')} | knew before {g.knew.mean() * 100:5.1f}")
            g = t[t.cond == "rag"]
            if task != "Del":
                print(f"    rag, full path in top-15:     real {rate(g[g.base_full == True])}")
                print(f"    rag, full path NOT in top-15: real {rate(g[g.base_full == False])}")
                for h in (1, 2):
                    print(f"    {h}-hop real pass: rag {rate(g[g.hop == h])} | oracle {rate(t[(t.cond == 'oracle') & (t.hop == h)])}")
                print(f"    hop minus rag (paired by question): {rate(t[t.cond == 'hop'], 'real', g)}")
            print(f"    oracle minus rag: {rate(t[t.cond == 'oracle'], 'real', g)}")
        aft[["ep", "entity", "task", "hop", "cond", "correct", "knew", "real", "base_full", "hop_full"]].to_json(
            os.path.join(M.OUT, f"reader_{tag}_per_question.json"), orient="records", indent=1)


if __name__ == "__main__":
    s = sys.argv[1]
    if s in ("answer", "judge") and os.name == "nt":     # process-scoped keep-awake, released on exit; no system setting changes
        import ctypes
        ctypes.windll.kernel32.SetThreadExecutionState(0x80000000 | 0x00000001)
    dry() if s == "dry" else answer("--think" in sys.argv) if s == "answer" else judge() if s == "judge" else stats()
