"""P2 BEAM knowledge update (CLOUD_NOTEBOOK.md pre-registration, 10-06): is an updated fact read by position when the
two conversations that state the old and the new value are shown chronological, reversed or in retrieval order?
BEAM (Mohammadta/BEAM, arXiv 2510.27246): 90 synthetic user-assistant conversations (100K / 500K / 1M tokens), sessions
dated by a time anchor, two knowledge_update questions per conversation with the ids of the old and the new message.
Frozen held-out set: print aggregates only.
Construction (mirrors explore_extmem.memconf): unit = one exchange (a user message and the assistant reply); a block =
neighbouring exchanges of one session, headed by that session's date. Old block = the old-evidence exchange +-1, new
block likewise; then, walking back from the new session, the middle 3 exchanges of every session in between, while the
prompt stays under ~20k tokens. The "->-> i,j" plan markers BEAM appends to messages are removed (they grow with time).
Judged by Qwen3-14B 4-bit with run_app_memory.JUDGE (A up-to-date / B outdated / C neither).
usage: python explore_beam.py check | run --models Qwen3-4B | judge | stats
"""
import argparse
import ast
import glob
import json
import os
import re
import zlib

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from paths import hub
from app_common import tag, JsonlAppender, chat_prompt, free_gpu, load_jsonl, load_reader
import run_app_memory as mem
from explore_extmem import orders
from explore_probe import LONG_SDPA, sdpa_kernel
from run_app_fix import generate

OUT = "results/beam_{}.jsonl"
REV = "bd579313ca79f5dfced05e37df92ae5ed6fa0b4a"   # refs/convert/parquet of Mohammadta/BEAM, downloaded 10-06
SPLITS = ("100K", "500K", "1M")
BUDGET_TOK, MAX_NEW, CONDS = 20000, 64, ("chrono", "rev", "retr")
MARK = re.compile(r"\s*->->\s*[\d,\s]+$")
_TOK = None


def ntok(s):
    global _TOK
    if _TOK is None:
        from transformers import AutoTokenizer
        _TOK = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
    return len(_TOK(s, add_special_tokens=False).input_ids)


def date(anchor):
    try:
        return pd.to_datetime(anchor, format="%B-%d-%Y").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return str(anchor)


def parse(s):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        return ast.literal_eval(s)


def exchanges(chat):
    """sessions -> exchanges [dict(sess, date, ids, text)]; a new exchange starts at every user message"""
    ex = []
    for si, sess in enumerate(chat):
        start = len(ex)
        for m in sess:
            line = ("User: " if m["role"] == "user" else "Assistant: ") + MARK.sub("", str(m["content"])).strip()
            if m["role"] == "user" or len(ex) == start:
                ex.append(dict(sess=si, date=date(m["time_anchor"]), ids=[], lines=[]))
            ex[-1]["ids"].append(int(m["id"]))
            ex[-1]["lines"].append(line)
    for e in ex:
        e["text"] = "\n".join(e.pop("lines"))
    return ex


def items(stats=None):
    out, stats = [], stats if stats is not None else {}
    bump = lambda k: stats.__setitem__(k, stats.get(k, 0) + 1)
    for sp in SPLITS:
        df = pd.read_parquet(hub("datasets--Mohammadta--BEAM", "snapshots", REV, "default", sp, "0000.parquet"))
        for r in df.itertuples():
            ex = exchanges(r.chat)
            where = {i: k for k, e in enumerate(ex) for i in e["ids"]}
            toks = {}
            T = lambda k: toks.setdefault(k, ntok(ex[k]["text"]))
            for qi, q in enumerate(parse(r.probing_questions).get("knowledge_update", [])):
                bump("questions")
                src = q.get("source_chat_ids") or {}
                old = [int(i) for i in src.get("original_info", []) if int(i) in where]
                new = [int(i) for i in src.get("updated_info", []) if int(i) in where]
                if not old or not new:
                    bump("excl_no_evidence_ids"); continue
                eo, en = where[old[0]], where[new[-1]]
                so, sn = ex[eo]["sess"], ex[en]["sess"]
                if so == sn:
                    bump("excl_same_session"); continue
                if so > sn or ex[eo]["date"] >= ex[en]["date"]:
                    bump("excl_old_not_earlier"); continue
                win = lambda k: [j for j in (k - 1, k, k + 1) if 0 <= j < len(ex) and ex[j]["sess"] == ex[k]["sess"]]
                blocks = {so: win(eo), sn: win(en)}
                tot = sum(T(j) for b in blocks.values() for j in b)
                if tot > BUDGET_TOK:
                    bump("excl_too_long"); continue
                for s in range(sn - 1, so, -1):   # sessions in between, newest first (as memconf)
                    ks = [k for k, e in enumerate(ex) if e["sess"] == s]
                    mid = ks[len(ks) // 2 - 1: len(ks) // 2 + 2] if len(ks) >= 3 else ks
                    t = sum(T(j) for j in mid)
                    if tot + t > BUDGET_TOK:
                        break
                    blocks[s] = mid; tot += t
                sess = sorted(blocks)
                docs = ["\n".join(ex[j]["text"] for j in blocks[s]) for s in sess]
                dates = [ex[blocks[s][0]]["date"] for s in sess]
                now = (pd.Timestamp(ex[en]["date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d") \
                    if re.fullmatch(r"\d{4}-\d\d-\d\d", ex[en]["date"]) else ex[en]["date"]
                refs = [str(x) for x in q.get("conversation_references") or []]
                msg = {int(m["id"]): MARK.sub("", str(m["content"])).strip() for s_ in r.chat for m in s_}
                old_ev, new_ev = (refs[0], refs[1]) if len(refs) == 2 else (msg[old[0]][:400], msg[new[-1]][:400])
                cover = all(where[i] in blocks[so] for i in old) and all(where[i] in blocks[sn] for i in new)
                bump("kept")
                qq = q["question"]
                for cond, idx in orders(qq, docs).items():
                    body = "\n\n".join(f"### Conversation (date: {dates[i]})\n{docs[i]}" for i in idx)
                    user = ("Here are records of your past conversations with the user.\n\n" + body +
                            f"\n\nCurrent date: {now}\nBased on the information above, answer the user's question in one "
                            f"short sentence.\nQuestion: {qq}")
                    out.append(dict(task="beam", id=f"{sp}-{r.conversation_id}-{qi}", conv=f"{sp}-{r.conversation_id}",
                                    split=sp, cond=cond, user=user, q=qq, ref=str(q.get("answer", "")), old=old_ev,
                                    new=new_ev, d_old=ex[eo]["date"], d_new=ex[en]["date"], n_blocks=len(docs),
                                    ctx_tok=tot, cover=cover, pos_new=idx.index(sess.index(sn))))
    return out


def run(args):
    its = items()
    for name in args.models.split(","):
        w = JsonlAppender(OUT.format(tag(name)), key=lambda r: (r["id"], r["cond"]))
        todo = [x for x in its if (x["id"], x["cond"]) not in w.done]
        print(name, "todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        for x in tqdm(todo, desc=name):
            p = chat_prompt(tok, x["user"])
            n = len(tok(p, add_special_tokens=False).input_ids)
            with sdpa_kernel(LONG_SDPA, set_priority=True):
                text = generate(tok, model, p, MAX_NEW)
            resp = text.rsplit("Answer:", 1)[1].strip().split("\n")[0] if "Answer:" in text else text.strip().split("\n")[0]
            w.write({k: v for k, v in x.items() if k != "user"} | dict(n_tok=n, response=resp, full=text))
            torch.cuda.empty_cache()
        w.close()
        del tok, model
        free_gpu()


@torch.no_grad()
def judge(args):
    tok = model = None
    for path in sorted(glob.glob(OUT.format("*"))):
        if path.endswith("_judged.jsonl"):
            continue
        w = JsonlAppender(path.replace(".jsonl", "_judged.jsonl"), key=lambda r: (r["id"], r["cond"]))
        todo = [r for r in load_jsonl(path) if (r["id"], r["cond"]) not in w.done]
        print(os.path.basename(path), "todo", len(todo), flush=True)
        if todo and model is None:
            tok, model = load_reader("Qwen3-14B")
        for r in tqdm(todo):
            text = mem.JUDGE.format(d0=r["d_old"], d1=r["d_new"], old=r["old"], new=r["new"], q=r["q"], a=r["ref"],
                                    r=r["response"])
            lp = mem.letter_logprobs(tok, model, text)
            lab = "ABC"[max(range(3), key=lambda i: lp[i])]
            w.write(r | dict(label=lab, lp=lp, correct=lab == "A", stale=lab == "B"))
        w.close()


def cluster_boot(df, a, b, n=10000, seed=0):
    """paired difference a - b (points), resampling whole conversations"""
    g = df.groupby("conv")
    conv = list(g.groups)
    sums = np.array([[g.get_group(c)[a].sum() - g.get_group(c)[b].sum(), len(g.get_group(c))] for c in conv])
    idx = np.random.default_rng(seed).integers(0, len(conv), (n, len(conv)))
    m = sums[idx, 0].sum(1) / sums[idx, 1].sum(1)
    return np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


def stats(args):
    from analyze_lora import boot
    for path in sorted(glob.glob(OUT.format("*").replace(".jsonl", "_judged.jsonl"))):
        df = pd.DataFrame(load_jsonl(path))
        name = os.path.basename(path)[len("beam_"):-len("_judged.jsonl")]
        piv = df.pivot_table(index="id", columns="cond", values="correct", aggfunc="first").dropna().astype(float)
        st = df.pivot_table(index="id", columns="cond", values="stale", aggfunc="first").astype(float).mean() * 100
        conv = df.drop_duplicates("id").set_index("id").conv
        print(f"========== {name} n={len(piv)} questions, {conv.nunique()} conversations | " +
              " ".join(f"{c} {piv[c].mean() * 100:5.1f}" for c in CONDS if c in piv) +
              " | stale " + " ".join(f"{c} {st.get(c, float('nan')):4.1f}" for c in CONDS))
        for c in ("rev", "retr"):
            if c in piv and "chrono" in piv:
                d, lo, hi, _ = boot(piv[c], piv.chrono)
                clo, chi = cluster_boot(piv.assign(conv=conv.loc[piv.index]), c, "chrono")
                print(f"      {c} - chrono {d:6.1f} [{lo:6.1f}, {hi:6.1f}] | conversation cluster [{clo:6.1f}, {chi:6.1f}]")


def check(args):
    st = {}
    its = items(st)
    df = pd.DataFrame([{k: v for k, v in x.items() if k not in ("user", "old", "new", "q", "ref")} for x in its])
    print("counts:", st)
    c = df[df.cond == "chrono"]
    print("questions", len(c), "| conversations", c.conv.nunique(), "| per split", c.split.value_counts().to_dict())
    print("context tokens min/med/max", int(c.ctx_tok.min()), int(c.ctx_tok.median()), int(c.ctx_tok.max()),
          "| blocks", c.n_blocks.value_counts().sort_index().to_dict(), "| all evidence ids inside blocks", round(c.cover.mean(), 3))
    print("position of the new block (0 = first shown):", df.groupby("cond").pos_new.mean().round(2).to_dict())
    print("plan markers left in prompts:", sum(bool(re.search(r"->->", x["user"])) for x in its))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "judge", "stats", "check"])
    ap.add_argument("--models", default="Qwen3-4B")
    args = ap.parse_args()
    {"run": run, "judge": judge, "stats": stats, "check": check}[args.stage](args)


if __name__ == "__main__":
    main()
