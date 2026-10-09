"""E19 (EXPLORE_PLAN.md, 10-09): synthetic multi-session chats, so that the dated-list procedure also triggers on
chat-memory prompts (E18.1 used it in only 6% of thoughts on the LongMemEval official prompt; E18 used it in 95% and
went from 43 to 73 on the items whose newest evidence is ranked first).

Everything here is generated: no benchmark content. Each sample = one user's chat sessions, each session headed by its
date; one attribute is stated 2-5 times with different values in different sessions (same symmetric statement
templates for every value, no recency cue words), other attributes and wiki-sentence small talk fill the sessions.
Sessions are shown chronological, reverse, shuffled, or "topic first" (sessions that mention the asked attribute
first, as a relevance-ranked retriever would). Chat formats: "Session k (date)" with User/Assistant lines, "[date] user:"
logs, markdown headings. The LongMemEval JSON history format is NOT used (kept as an unseen format).
Questions: current value / first value / most recent change -> the thought lists the dated records (E18 target);
reason for a change / attribute never mentioned -> short thought without a list. Thinking on / off 1/2 each.
usage: python gen_chat_data.py   -> data_train/chat_{train,val,dev}.jsonl
"""
import collections
import datetime as dt
import hashlib
import json
import random

import gen_train_data as g
from gen_bind_data_v2 import cap
from gen_bind_data_v4 import REASONS

STATE = ["For my notes, {a} is {v}.", "Just so you have it: {a} is {v}.", "Writing this down - {a} is {v}.",
         "Please remember that {a} is {v}.", "One thing for you to keep: {a} is {v}.", "{a} is {v}, by the way."]
ACK = ["Got it, I'll keep that in mind.", "Noted.", "Thanks, I've saved that.", "Understood, I'll remember it."]
FORMATS = {"session": "Session {k} ({d})", "log": None, "md": "#### Chat on {d}"}
INTRO = {"train": ["Here are my past chat sessions with you.", "Below are our earlier conversations.",
                   "These are records of the chats we had before."],
         "dev": ["What follows are the previous conversations between the user and you."]}
DFMT = ["%Y-%m-%d", "%b %d, %Y", "%d %B %Y", "%A, %d %b %Y"]
QS = {"train": {"current": "Based on our chats, what is {a} as of the last conversation?",
                "first": "Based on our chats, what was {a} the first time it came up?",
                "change": "Based on our chats, what was the most recent change to {a}?",
                "reason": "Based on our chats, why was {a} set to {v}?",
                "absent": "Based on our chats, what is {a}?"},
      "dev": {"current": "Looking at the conversations, what is {a} at the time of the latest one?",
              "first": "Looking at the conversations, what was {a} when it was first mentioned?",
              "change": "Looking at the conversations, how did {a} change most recently?",
              "reason": "Looking at the conversations, what was the reason {a} became {v}?",
              "absent": "Looking at the conversations, what is {a}?"}}
INSTR = {"train": ["Answer in one short sentence.", "Answer with the value only.", "Reply in one complete sentence."],
         "dev": ["Respond in a single sentence."]}
KINDS, KW = ["current", "first", "change", "reason", "absent"], [0.40, 0.12, 0.18, 0.18, 0.12]


def small_talk(rng, wiki):
    return [("User", g.wiki_sentences(rng, wiki, rng.randint(15, 40))), ("Assistant", g.wiki_sentences(rng, wiki, rng.randint(20, 60)))]


def render(fmt, sessions, order):
    out = []
    for k, i in enumerate(order, 1):
        d, turns = sessions[i]
        if fmt == "log":
            out.append("\n".join(f"[{d}] {who.lower()}: {txt}" for who, txt in turns))
        else:
            head = FORMATS[fmt].format(k=k, d=d)
            out.append(head + "\n" + "\n".join(f"{who}: {txt}" for who, txt in turns))
    return "\n\n".join(out)


def make(rng, wiki, split):
    sp = "dev" if split == "dev" else "train"
    keys = rng.sample(sorted(g.ATTRS), 4)
    target, others = keys[0], keys[1:]
    a = g.ATTRS[target][0]
    n_val = rng.randint(2, 5)
    vals = []
    while len(vals) < n_val:
        v = g.ATTRS[target][1](rng)
        if v not in vals:
            vals.append(v)
    n_sess = n_val + rng.randint(1, 4)
    t_sess = sorted(rng.sample(range(n_sess), n_val))
    kind = rng.choices(KINDS, KW)[0]
    k_reason = rng.randrange(1, n_val)
    reason = rng.choice(REASONS[sp])
    fmt = rng.choice(list(FORMATS))
    dfmt = rng.choice(DFMT)
    t0 = dt.date(rng.randint(2016, 2025), rng.randint(1, 12), rng.randint(1, 28))
    dates = []
    for _ in range(n_sess):
        dates.append(t0)
        t0 += dt.timedelta(days=rng.randint(3, 75))
    ds = [x.strftime(dfmt) for x in dates]
    sessions = []
    for s in range(n_sess):
        turns = []
        for _ in range(rng.randint(1, 3)):
            turns += small_talk(rng, wiki)
        if s in t_sess:
            vi = t_sess.index(s)
            line = rng.choice(STATE).format(a=a, v=vals[vi])
            if kind == "reason" and vi == k_reason:
                line = line.rstrip(".") + f", because {reason}."
            turns.insert(rng.randrange(0, len(turns) + 1, 2), ("User", cap(line)))
            turns.insert(turns.index(("User", cap(line))) + 1, ("Assistant", rng.choice(ACK)))
        for o in others:
            if rng.random() < 0.35:
                turns.insert(rng.randrange(0, len(turns) + 1, 2), ("User", cap(rng.choice(STATE).format(a=g.ATTRS[o][0], v=g.ATTRS[o][1](rng)))))
        sessions.append((ds[s], turns))
    order_kind = rng.choice(["chrono", "reverse", "shuffle", "topic_first"])
    idx = list(range(n_sess))
    if order_kind == "reverse":
        idx = idx[::-1]
    elif order_kind == "shuffle":
        rng.shuffle(idx)
    elif order_kind == "topic_first":
        on = [i for i in idx if i in t_sess]
        off = [i for i in idx if i not in t_sess]
        rng.shuffle(on); rng.shuffle(off)
        idx = on + off
    chain = [(ds[s], vals[t_sess.index(s)]) for s in t_sess]
    lst = [f"Records about {a}, from oldest to newest:"] + [f"- {d}: {v}" for d, v in chain]
    instr = rng.choice(INSTR[sp])
    if kind == "absent":
        a_q = g.ATTRS[rng.choice(sorted(set(g.ATTRS) - set(keys)))][0]
        q = QS[sp]["absent"].format(a=a_q)
        thought, final = f"No record mentions {a_q}.", ("Not mentioned" if "value only" in instr else f"Our chats do not mention {a_q}.")
    elif kind == "reason":
        q = QS[sp]["reason"].format(a=a, v=vals[k_reason])
        thought = (f"The question asks for the reason behind a change, not for the current value.\n"
                   f"In the session of {chain[k_reason][0]}, {a} was set to {vals[k_reason]} because {reason}.")
        final = f"It was set to {vals[k_reason]} because {reason}."
        instr = rng.choice([x for x in INSTR[sp] if "value only" not in x] or INSTR[sp])
    elif kind == "current":
        q = QS[sp]["current"].format(a=a)
        thought = "\n".join(lst + [f"The newest record says {vals[-1]}."])
        final = vals[-1] if "value only" in instr else f"As of the last conversation, {a} is {vals[-1]}."
    elif kind == "first":
        q = QS[sp]["first"].format(a=a)
        thought = "\n".join(lst + [f"The oldest record says {vals[0]}."])
        final = vals[0] if "value only" in instr else f"The first time it came up, {a} was {vals[0]}."
    else:
        q = QS[sp]["change"].format(a=a)
        thought = "\n".join(lst + [f"The last two records say {vals[-2]} and then {vals[-1]}."])
        final = f"{cap(a)} changed from {vals[-2]} to {vals[-1]}."
        instr = rng.choice([x for x in INSTR[sp] if "value only" not in x] or INSTR[sp])
    prompt = rng.choice(INTRO[sp]) + "\n\n" + render(fmt, sessions, idx) + f"\n\n{q} {instr}"
    think = rng.random() < 0.5
    answer = f"<think>\n{thought}\n</think>\n\n{final}" if think else final
    return dict(kind=f"chat_{kind}", qtype=f"chat_{kind}", fmt=f"chat_{fmt}", order=order_kind, dated=True, think=think,
                prompt=prompt, answer=answer, final=final, n_sess=n_sess, n_val=n_val)


def main():
    wiki = g.load_wiki()
    for split, n, seed in (("train", 1000, 1919), ("val", 100, 1920), ("dev", 100, 1921)):
        rng = random.Random(seed)
        rows = [dict(make(rng, wiki, split), id=f"chat-{split}-{i}") for i in range(n)]
        path = f"data_train/chat_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(path, len(rows), hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16].upper())
        print("  kind", dict(collections.Counter(r["kind"] for r in rows)), "| fmt", dict(collections.Counter(r["fmt"] for r in rows)),
              "| order", dict(collections.Counter(r["order"] for r in rows)), "| think", sum(r["think"] for r in rows))
        cue = sum(bool(g.CUE.search(r["prompt"].split("\n\n", 1)[1].rsplit("\n\n", 1)[0])) for r in rows)
        vis = sum("from oldest to newest" in r["answer"].split("</think>")[-1] for r in rows)
        print("  prompts whose chat text contains a recency cue word:", cue, "| visible replies with a list:", vis)


if __name__ == "__main__":
    main()
