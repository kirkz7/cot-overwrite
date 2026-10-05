"""External memory benchmarks built by others (10-04, EXPLORE_PLAN.md "外部测试集"). BASE models only, no training.
Each set is shown in three orders of the SAME units (only the order differs):
  chrono  oldest first (source order)
  rev     newest first
  retr    BM25 relevance to the question, most relevant first (what a standard retrieval memory would hand the model)
Sets:
  memconf  MemConflict (TaoZhen1110/MemConflict, Data/Step4_4.jsonl): dynamic-conflict "what changed" questions (medium).
           Kept only when the old value is NOT mentioned in the update session but IS in an earlier session (structural
           filter, no model output used). Context = that earlier session + the update session + the sessions just before
           the update, up to ~20k source tokens; each session headed by its date. Judged by Qwen3-14B (A right direction,
           B reversed / old value as new, C neither).
  mabcr    MemoryAgentBench Conflict_Resolution, factconsolidation_sh_6k: numbered facts, larger number = newer.
           NOT held-out: already a pre-registered LoRA test set and used in E7; the LoRA data has its format.
           Primary subset: the questions whose answer fact has exactly one older conflicting fact. String-matched.
           (sh_32k is 37.5k tokens: does not fit 16 GB; later.)
  locomo   LoCoMo (snap-research/locomo, locomo10.json): all temporal questions (category 2), full conversations,
           each session headed by its date. Judged by Qwen3-14B (A correct / B incorrect).
No conversation text is printed.
usage: python explore_extmem.py run --models Qwen3-4B | judge | stats | check
"""
import argparse
import glob
import json
import math
import os
import random
import re
from collections import Counter

import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, JsonlAppender, chat_prompt, free_gpu, load_jsonl, load_reader, answer_text
import run_app_memory as mem
from explore_probe import LONG_SDPA, sdpa_kernel
from run_app_fix import generate

OUT = "results/extmem_{}.jsonl"
MC_PATH = r"D:\datasets\memconflict\Step4_4.jsonl"
LOCOMO_PATH = r"D:\datasets\locomo\locomo10.json"
MAB_GLOB = r"D:\hf_cache\hub\datasets--ai-hyz--MemoryAgentBench\snapshots\*\data\Conflict_Resolution-*.parquet"
CONDS = ("chrono", "rev", "retr")
JUDGED = {"v1": "_judged.jsonl", "v2": "_judged2.jsonl"}   # parse rule v1: first line after "Answer:"; v2: app_common.answer_text
BUDGET = {"memconf": 64, "memconf_rs": 64, "mabcr": 24, "locomo": 32}


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def bm25_order(query, docs, k1=1.5, b=0.75):
    """indices of docs sorted by BM25 score for query, best first (ties: original order)"""
    toks = [words(d) for d in docs]
    avg = sum(map(len, toks)) / max(1, len(toks))
    df = Counter(w for t in toks for w in set(t))
    n = len(docs)
    q = set(words(query))
    scores = []
    for t in toks:
        tf = Counter(t)
        s = 0.0
        for w in q:
            if tf[w]:
                idf = math.log(1 + (n - df[w] + 0.5) / (df[w] + 0.5))
                s += idf * tf[w] * (k1 + 1) / (tf[w] + k1 * (1 - b + b * len(t) / avg))
        scores.append(s)
    return sorted(range(n), key=lambda i: (-scores[i], i))


def orders(query, docs):
    idx = list(range(len(docs)))
    return {"chrono": idx, "rev": idx[::-1], "retr": bm25_order(query, docs)}


# ---------------------------------------------------------------- MemConflict
def key(v):
    return str(v).split(",")[0].strip().lower()


def flat_pairs(upd):
    out = []
    for u in upd or []:
        b, a = u["Before"], u["After"]
        if isinstance(b, dict) and isinstance(a, dict):
            out += [(f"{u['Attribute']}.{k}", str(b[k]), str(a[k])) for k in b if k in a and isinstance(b[k], str) and b[k] != a[k]]
        elif isinstance(b, str) and isinstance(a, str) and b != a:
            out.append((u["Attribute"], b, a))
    return out


def sess_text(s):
    turns = sorted(s["Session_Dialogue"].items(), key=lambda kv: int(kv[0].rsplit("_", 1)[1]))
    def line(m):  # 28 of 142k messages are malformed ({"assistant": text}, no content, ...)
        role = m.get("role") or ("assistant" if "assistant" in m else "user")
        text = m.get("content") or m.get("assistant") or ""
        return f"{'User' if role == 'user' else 'Assistant'}: {text}" if text else None
    return "\n".join(l for _, t in turns for m in t if (l := line(m)))


def memconf(max_per=8, budget=20000, restated=False, task="memconf"):
    """restated=True (task memconf_rs, review 10-04): the complementary items whose update session itself restates the
    old value ("moved from A to B"), so the direction of the change is written in one session (boundary control)"""
    out = []
    for inst in map(json.loads, open(MC_PATH, encoding="utf-8")):
        ch = inst["Full_Session_Chain"]
        txt = [sess_text(s).lower() for s in ch]
        cand = []
        for ui, U in enumerate(ch):
            for q in U["Session_Questions"]:
                if q["conflict_type"] != "dynamic_conflict" or q["difficulty"] != "medium":
                    continue
                ans = q["answer"].lower()
                m = [p for p in flat_pairs(U.get("Updated_Attributes")) if key(p[1]) in ans and key(p[2]) in ans]
                if len(m) != 1 or len(key(m[0][1])) < 3:
                    continue
                attr, b, a = m[0]
                if (key(b) in txt[ui]) != restated:
                    continue
                prev = [i for i in range(ui) if key(b) in txt[i]]
                if not prev:
                    continue
                bi = prev[-1]
                L = lambda i: ch[i]["Session_Dialogue_Token_Length"]
                keep, tot = {bi, ui}, L(bi) + L(ui)
                if tot > budget:
                    continue
                for i in range(ui - 1, bi, -1):
                    if tot + L(i) > budget:
                        break
                    keep.add(i); tot += L(i)
                cand.append(dict(inst=inst["ID"], ui=ui, q=q, attr=attr, b=b, a=a, sess=sorted(keep)))
        random.Random(inst["ID"]).shuffle(cand)
        for c in cand[:max_per]:
            ch_sel = [ch[i] for i in c["sess"]]
            docs = [sess_text(s) for s in ch_sel]
            dates = [s["Date"] for s in ch_sel]
            now = (pd.Timestamp(ch[c["ui"]]["Date"]) + pd.Timedelta(days=1)).strftime("%Y-%m-%d")
            q = c["q"]["question"]
            for cond, idx in orders(q, docs).items():
                blocks = [f"### Conversation (date: {dates[i]})\n{docs[i]}" for i in idx]
                user = ("Here are records of your past conversations with the user.\n\n" + "\n\n".join(blocks) +
                        f"\n\nCurrent date: {now}\nBased on the information above, answer the user's question in one short "
                        f"sentence.\nQuestion: {q}")
                out.append(dict(task=task, id=f"{c['inst'][:8]}-{c['ui']}-{c['q']['question_id']}", cond=cond, user=user,
                                q=q, ref=c["q"]["answer"], old=c["b"], new=c["a"], attr=c["attr"],
                                d_old=ch[[i for i in c["sess"] if i < c["ui"] and key(c["b"]) in txt[i]][-1]]["Date"],
                                d_new=ch[c["ui"]]["Date"], n_sess=len(docs),
                                pos_new={k: v.index(len(docs) - 1) for k, v in orders(q, docs).items()}[cond]))
    return out


# ---------------------------------------------------------------- MemoryAgentBench CR
def norm(s):
    return " ".join(words(s))


def mabcr(source="factconsolidation_sh_6k"):
    df = pd.read_parquet(glob.glob(MAB_GLOB)[0])
    r = df[df.metadata.apply(lambda m: m["source"]) == source].iloc[0]
    facts = [(int(n), f) for n, f in (re.match(r"(\d+)\. (.*)", l).groups() for l in r.context.split("\n") if re.match(r"\d+\. ", l))]
    docs = [f"{n}. {f}" for n, f in facts]
    out = []
    for k, (q, a) in enumerate(zip(r.questions, r.answers)):
        gold = [str(x) for x in a]
        g = gold[0]
        hits = [(n, f) for n, f in facts if f.rstrip(".").endswith(g)]
        old, conflict = None, False
        if hits:
            n, f = max(hits)
            pre = f[: f.rstrip(".").rfind(g)]
            olds = [(m, h) for m, h in facts if len(pre) > 10 and h.startswith(pre) and m != n]
            if len(olds) == 1 and olds[0][0] < n:
                conflict, old = True, olds[0][1][len(pre):].rstrip(".")
        for cond, idx in orders(q, docs).items():
            user = ("Here is a list of facts. Each fact starts with a serial number; a fact with a larger serial number is "
                    "newer and replaces any older fact it conflicts with.\n\n" + "\n".join(docs[i] for i in idx) +
                    f"\n\nAnswer the question using the newest facts. Reply with the answer entity only.\nQuestion: {q}")
            out.append(dict(task="mabcr", id=f"{source}-{k}", cond=cond, user=user, q=q, gold=gold, old=old, conflict=conflict))
    return out


# ---------------------------------------------------------------- LoCoMo
def locomo():
    out = []
    for x in json.load(open(LOCOMO_PATH, encoding="utf-8")):
        c = x["conversation"]
        ks = sorted([k for k in c if re.fullmatch(r"session_\d+", k)], key=lambda k: int(k.split("_")[1]))
        def line(m):
            cap = f" [shares a photo: {m['blip_caption']}]" if m.get("blip_caption") else ""
            return f"{m['speaker']}: {m['text']}{cap}"
        docs = ["\n".join(line(m) for m in c[k]) for k in ks]
        dates = [c[f"{k}_date_time"] for k in ks]
        for j, qa in enumerate(x["qa"]):
            if qa["category"] != 2:
                continue
            for cond, idx in orders(qa["question"], docs).items():
                blocks = [f"### Conversation (date: {dates[i]})\n{docs[i]}" for i in idx]
                user = (f"Here are records of past conversations between {c['speaker_a']} and {c['speaker_b']}.\n\n" +
                        "\n\n".join(blocks) + "\n\nBased on the conversations above, answer the question in one short "
                        f"phrase.\nQuestion: {qa['question']}")
                out.append(dict(task="locomo", id=f"{x['sample_id']}-{j}", cond=cond, user=user, q=qa["question"],
                                ref=str(qa["answer"])))
    return out


def items(tasks=""):
    """default: the three frozen sets; memconf_rs (restated-old-value control) only when asked for explicitly"""
    out = mabcr() + memconf() + locomo()
    if "memconf_rs" in tasks.split(","):
        out += memconf(restated=True, task="memconf_rs")
    return out


@torch.no_grad()
def run(args):
    its = [x for x in items(args.tasks) if not args.tasks or x["task"] in args.tasks.split(",")]
    for name in args.models.split(","):
        w = JsonlAppender(OUT.format(tag(name)), key=lambda r: (r["task"], r["id"], r["cond"]))
        todo = [x for x in its if (x["task"], x["id"], x["cond"]) not in w.done]
        print(name, "todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        for x in tqdm(todo, desc=name):
            p = chat_prompt(tok, x["user"])
            n = len(tok(p, add_special_tokens=False).input_ids)
            with sdpa_kernel(LONG_SDPA, set_priority=True):
                text = generate(tok, model, p, args.budget or BUDGET[x["task"]])
            # reasoning-trained readers (E13): the answer is the text after the last "Answer:"; others: the first line
            resp = text.rsplit("Answer:", 1)[1].strip().split("\n")[0] if "Answer:" in text else text.strip().split("\n")[0]
            rec = {k: v for k, v in x.items() if k != "user"}
            rec.update(n_tok=n, response=resp, full=text)
            if x["task"] == "mabcr":
                hit = any(norm(g) in norm(resp) for g in x["gold"])
                rec.update(correct=hit, stale=bool(x["old"]) and not hit and norm(x["old"]) in norm(resp))
            w.write(rec)
            torch.cuda.empty_cache()
        w.close()
        del tok, model
        free_gpu()


MC_JUDGE = """You are grading an answer about how a user's information changed over time.
Earlier value (from an older conversation, {d0}): "{old}"
Later value (from a newer conversation, {d1}): "{new}"

Question: {q}
Reference answer: {ref}
Response to grade: {r}

Which option best describes the response?
A. It describes the change in the correct direction (from the earlier value to the later value), or gives the later value as the current one.
B. It gets the direction reversed (from the later value to the earlier one), or gives the earlier value as the current or new one.
C. Neither (wrong values, unrelated, or says it does not know).
Reply with a single letter: A, B, or C."""

LC_JUDGE = """You are grading an answer to a question about past conversations.
Question: {q}
Reference answer: {ref}
Response to grade: {r}

Does the response give the same answer as the reference? Dates may be written differently; an answer with the same meaning counts as correct.
A. Correct
B. Incorrect
Reply with a single letter: A or B."""


@torch.no_grad()
def judge(args):
    tok = model = None
    for path in sorted(glob.glob(OUT.format("*"))):
        if "_judged" in path:
            continue
        rows = [r for r in load_jsonl(path) if r["task"] in ("memconf", "memconf_rs", "locomo")]
        if args.parse == "v2":       # E17 pre-registration: the judge sees all text after the last "Answer:"
            rows = [r | {"response": answer_text(r["full"])} for r in rows if r["task"] in ("memconf", "memconf_rs")]
        w = JsonlAppender(path.replace(".jsonl", JUDGED[args.parse]), key=lambda r: (r["task"], r["id"], r["cond"]))
        todo = [r for r in rows if (r["task"], r["id"], r["cond"]) not in w.done]
        print(os.path.basename(path), "judge todo", len(todo), flush=True)
        if todo and model is None:
            tok, model = load_reader("Qwen3-14B")
        for r in tqdm(todo):
            if r["task"] in ("memconf", "memconf_rs"):
                text = MC_JUDGE.format(d0=r["d_old"], d1=r["d_new"], old=r["old"], new=r["new"], q=r["q"], ref=r["ref"], r=r["response"])
                lp = mem.letter_logprobs(tok, model, text)
                lab = "ABC"[max(range(3), key=lambda i: lp[i])]
            else:
                text = LC_JUDGE.format(q=r["q"], ref=r["ref"], r=r["response"])
                lp = mem.letter_logprobs(tok, model, text)
                lab = "AB"[max(range(2), key=lambda i: lp[i])]
            w.write(dict(task=r["task"], id=r["id"], cond=r["cond"], label=lab, correct=lab == "A", stale=lab == "B" and r["task"] in ("memconf", "memconf_rs")))
        w.close()


def stats(args):
    from analyze_lora import boot
    for path in sorted(glob.glob(OUT.format("*"))):
        if "_judged" in path:
            continue
        name = os.path.basename(path)[len("extmem_"):-len(".jsonl")]
        raw = pd.DataFrame(load_jsonl(path))
        jp = path.replace(".jsonl", JUDGED[args.parse])
        if os.path.exists(jp):
            j = pd.DataFrame(load_jsonl(jp)).set_index(["task", "id", "cond"])
            raw = raw.set_index(["task", "id", "cond"])
            for col in ("correct", "stale"):
                if col in j:
                    raw[col] = raw[col].where(raw[col].notna(), j[col]) if col in raw else j[col]
            raw = raw.reset_index()
        print("=" * 10, name)
        for task, g in raw.groupby("task"):
            subsets = {"all": g}
            if task == "mabcr":
                subsets = {"conflict (primary)": g[g.conflict == True], "no conflict": g[g.conflict == False]}
            for sname, h in subsets.items():
                h = h[h.correct.notna()]
                if not len(h):
                    continue
                piv = h.pivot_table(index="id", columns="cond", values="correct", aggfunc="first").dropna().astype(float)
                st = h.pivot_table(index="id", columns="cond", values="stale", aggfunc="first").astype(float).mean() * 100
                line = f"{task:8s} {sname:18s} n={len(piv):3d} | " + " ".join(f"{c} {piv[c].mean() * 100:5.1f}" for c in CONDS if c in piv)
                line += " | stale " + " ".join(f"{c} {st.get(c, float('nan')):4.1f}" for c in CONDS)
                print(line + f" | median tok {int(h.n_tok.median())}")
                for c in ("rev", "retr"):
                    if c in piv and "chrono" in piv:
                        d, lo, hi, _ = boot(piv[c], piv.chrono)
                        print(f"      {c} - chrono {d:6.1f} [{lo:6.1f}, {hi:6.1f}]")


def check(args):
    from transformers import AutoTokenizer
    from app_common import MODELS
    tok = AutoTokenizer.from_pretrained(MODELS["Qwen3-4B"][0])
    its = items()
    df = pd.DataFrame([{k: v for k, v in x.items() if k != "user"} | {"n_tok": len(tok(x["user"]).input_ids)} for x in its])
    for task, g in df.groupby("task"):
        print(task, "questions", g.id.nunique(), "| tokens min/med/max", g.n_tok.min(), int(g.n_tok.median()), g.n_tok.max())
    m = df[(df.task == "memconf")]
    if len(m):
        print("memconf sessions per item", m.n_sess.describe()[["min", "50%", "max"]].tolist(), "| attrs", m[m.cond == "chrono"].attr.str.split(".").str[0].value_counts().to_dict())
        print("memconf position of the update session (0 = first shown):", m.groupby("cond").pos_new.mean().round(2).to_dict())
    a = df[(df.task == "mabcr") & (df.cond == "chrono")]
    print("mabcr conflict items", int(a.conflict.sum()), "/", len(a))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "judge", "stats", "check"])
    ap.add_argument("--models", default="Qwen3-4B,Phi-4-mini")
    ap.add_argument("--tasks", default="", help="comma list of memconf,mabcr,locomo (default: all)")
    ap.add_argument("--parse", choices=["v1", "v2"], default="v1", help="judge / stats: answer parse rule (v2 from E17 on)")
    ap.add_argument("--budget", type=int, default=None, help="new tokens; default per task (64/24/32); E13 readers: 320")
    args = ap.parse_args()
    {"run": run, "judge": judge, "stats": stats, "check": check}[args.stage](args)


if __name__ == "__main__":
    main()
