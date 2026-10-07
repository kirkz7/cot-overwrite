"""Mitigation baselines for order-as-time errors on LongMemEval knowledge-update items.

Same 78 items and judge as run_app_memory.py. Every condition shows dates; the order is either
chronological (chrono / newlast) or newest-first (rev / oldlast), so each fix is measured on the failure
case and checked for harm on the chronological case.
  instr     instruction: the records may not be in chronological order; use the most recent date
  timeline  list the relevant statements with their dates first, then answer from the most recent
  qfirst    question placed BEFORE the records (control: proximity to the question vs order of the updates)
  think     Qwen3 thinking mode (Qwen3-4B; sampled, 2048-token budget, then the official early-exit sentence)
Stages: read (instr/timeline/qfirst, all readers) | think (Qwen3-4B) | judge
usage: python run_app_fix.py read | think | judge
"""
import argparse
import glob
import os
import re

import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, JsonlAppender, chat_prompt, free_gpu, load_jsonl, load_reader
from probe import greedy
from run_app_memory import JUDGE, N_DISTRACT, clip, letter_logprobs, load_items

ORDERS_S = ["chrono", "rev"]
ORDERS_T = ["newlast", "oldlast"]
FIXES = ["instr", "timeline", "qfirst"]
INSTR = ("Note: the records are not necessarily in chronological order. If the user's information changed "
         "over time, rely on the record with the most recent date.\n")
TIMELINE = ("First, list every statement in the records that is relevant to the question, each with its date. "
            "Then answer using the most recent statement, on a final line that starts with \"Answer:\".\n")
EARLY_EXIT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now."


def records(it, cond):
    """The dated records in the order given by cond (S_* = two sessions, T_* = retrieved user turns)."""
    if cond.startswith("S_"):
        order = [0, 1] if "_chrono_" in cond else [1, 0]
        blocks = [f"### Session (date: {it['dates'][i]})\n" + "\n".join(f"{t['role']}: {t['content']}" for t in it["sessions"][i])
                  for i in order]
        return "Here are records of your past chat sessions with the user.\n\n" + "\n\n".join(blocks)
    old, new = (0, it["old_ev"]), (1, it["new_ev"])
    first, second = (old, new) if "_newlast_" in cond else (new, old)
    mems, di = [], iter(it["distract"])
    for pos in range(N_DISTRACT + 2):
        if pos == it["slots"][0]:
            mems.append(first)
        elif pos == it["slots"][1]:
            mems.append(second)
        else:
            s, txt = next(di, (None, None))
            if s is not None:
                mems.append((s, clip(txt)))
    return "Relevant memories retrieved from past conversations with the user:\n" + \
        "\n".join(f"- ({it['dates'][s]}) The user said: \"{txt}\"" for s, txt in mems)


def build(it, cond):
    fix = cond.split("_")[-1]
    body = records(it, cond)
    date = f"Current date: {it['qdate']}\n"
    if fix == "qfirst":
        return f"The user asks: {it['question']}\n\n{body}\n\n{date}Answer the user's question above in one short sentence."
    ask = f"Question: {it['question']}"
    if fix == "instr":
        return f"{body}\n\n{date}{INSTR}Based on the information above, answer the user's question in one short sentence.\n{ask}"
    if fix == "timeline":
        return f"{body}\n\n{date}{TIMELINE}{ask}"
    return f"{body}\n\n{date}Based on the information above, answer the user's question in one short sentence.\n{ask}"   # think


def conds_for(it, s_level=True):
    out = []
    if s_level:
        out += [f"S_{o}_{f}" for o in ORDERS_S for f in FIXES]
    if it["has_ev"]:
        out += [f"T_{o}_{f}" for o in ORDERS_T for f in FIXES]
    return out


_STOPS = {}


def stop_ids(tok):
    """End-of-turn ids of this tokenizer (only tokens that really exist in its vocabulary)."""
    if id(tok) not in _STOPS:
        vocab = tok.get_vocab()
        _STOPS[id(tok)] = {tok.eos_token_id} | {vocab[t] for t in ("<|im_end|>", "<|end|>", "<|eot_id|>", "<|endoftext|>", "<end_of_turn>")
                                                if t in vocab}
    return _STOPS[id(tok)]


def generate(tok, model, prompt, max_new):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    return tok.decode(greedy(model, ids, max_new, stop_ids(tok)), skip_special_tokens=True).strip()


QWEN_THINK_SAMPLING = dict(do_sample=True, temperature=0.6, top_p=0.95, top_k=20)   # Qwen3 model card, thinking mode


def generate_think(tok, model, prompt, max_new, key):
    """E18 thinking mode, decoded as the Qwen3 model card recommends for thinking (greedy decoding there "can lead to
    endless repetitions"; seen on PersonaMem 10-06). Seeded from the item key, so every run is reproducible."""
    import zlib
    if getattr(model, "is_vllm", False):   # cloud: the same sampling on a vLLM server, seeded per item
        kw = {k: v for k, v in QWEN_THINK_SAMPLING.items() if k != "do_sample"}
        ids = tok(prompt, add_special_tokens=False).input_ids
        out = model.sample(ids, max_new, stop_ids(tok), zlib.crc32(str(key).encode()), **kw)
        return tok.decode(out, skip_special_tokens=True).strip()
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    torch.manual_seed(zlib.crc32(str(key).encode()))
    out = model.generate(ids, attention_mask=torch.ones_like(ids), max_new_tokens=max_new, eos_token_id=sorted(stop_ids(tok)),
                         pad_token_id=tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id,
                         **QWEN_THINK_SAMPLING)
    return tok.decode(out[0, ids.shape[1]:], skip_special_tokens=True).strip()


def final_answer(cond, text):
    """The part of a response that the judge grades."""
    if cond.endswith("timeline"):
        m = re.findall(r"Answer:\s*(.+)", text)
        return m[-1].strip() if m else text[-300:]
    return text.strip().split("\n")[0] if not cond.endswith("think") else text.strip()[:400]


def read(args):
    items = load_items()[:args.limit]
    for name in args.models.split(","):
        w = JsonlAppender(f"results/app_fix_{tag(name)}.jsonl", key=lambda r: (r["qid"], r["cond"]))
        jobs = [(it, c) for it in items for c in conds_for(it)]
        todo = [(it, c) for it, c in jobs if (it["qid"], c) not in w.done]
        print(name, "done", len(jobs) - len(todo), "todo", len(todo), flush=True)
        if todo:
            tok, model = load_reader(name)
            ctx = getattr(model.config, "max_position_embeddings", 32768)
            for it, cond in tqdm(todo, desc=name):
                p = chat_prompt(tok, build(it, cond))
                n = len(tok(p, add_special_tokens=False).input_ids)
                max_new = 320 if cond.endswith("timeline") else 64
                if n + max_new > ctx:
                    w.write(dict(model=name, qid=it["qid"], cond=cond, n_tok=n, skipped=True))
                    continue
                text = generate(tok, model, p, max_new)
                w.write(dict(model=name, qid=it["qid"], cond=cond, n_tok=n, skipped=False, full=text,
                             response=final_answer(cond, text)))
                if n > 6000:
                    torch.cuda.empty_cache()
            del tok, model
            free_gpu()
        w.close()


def think(args):
    """Qwen3-4B thinking mode on the dated chrono / newest-first conditions (S and T level)."""
    from fastgen import GraphGen, auto_batch
    items = load_items()[:args.limit]
    name = "Qwen3-4B"
    w = JsonlAppender(f"results/app_fix_{name}_think.jsonl", key=lambda r: (r["qid"], r["cond"]))
    jobs = [(it, f"S_{o}_think") for it in items for o in ORDERS_S] + \
           [(it, f"T_{o}_think") for it in items if it["has_ev"] for o in ORDERS_T]
    todo = [(it, c) for it, c in jobs if (it["qid"], c) not in w.done]
    print(name, "think done", len(jobs) - len(todo), "todo", len(todo), flush=True)
    if todo:
        tok, model = load_reader(name)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        def head(text):
            return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True).split("<think>")[0] + "<think>\n"
        todo.sort(key=lambda j: len(build(*j)))                     # similar lengths per batch: less padding
        enc = [tok(head(build(it, c)), add_special_tokens=False).input_ids for it, c in todo]
        L = max(map(len, enc)) + args.budget + 256                # room for the early-exit sentence + a short answer
        bs = auto_batch(model, L, cap=8)
        print("batch size", bs, "cache length", L, flush=True)
        gg = GraphGen(model, bs, L)
        for b in tqdm(range(0, len(todo), bs)):
            g = torch.Generator(device="cuda").manual_seed(1000 + b)
            out = gg.generate(enc[b:b + bs], args.budget, {im_end}, tok.pad_token_id, sample=True, generator=g)
            recs = []
            for (it, cond), ids, (row, stop) in zip(todo[b:b + bs], enc[b:b + bs], out):
                text = tok.decode(row, skip_special_tokens=False)
                if "</think>" in text:
                    thought, ans = text.split("</think>", 1)
                    ans = ans.replace("<|im_end|>", "").strip()
                    finished = True
                else:                                           # budget hit: official early exit, then a short answer
                    thought, finished = text, False
                    p2 = tok.decode(ids, skip_special_tokens=False) + text + EARLY_EXIT + "\n</think>\n\n"
                    ans = ""
                    for x in gg.generate([tok(p2, add_special_tokens=False).input_ids], 96, {im_end}, tok.pad_token_id,
                                         sample=False):
                        ans = tok.decode(x[0], skip_special_tokens=True).strip()
                recs.append(dict(model=name, qid=it["qid"], cond=cond, skipped=False, finished=finished,
                                 n_think=len(tok(thought, add_special_tokens=False).input_ids), full=ans,
                                 response=ans.split("\n")[0][:400]))
            w.write(*recs)
        del tok, model, gg
        free_gpu()
    w.close()


@torch.no_grad()
def judge(args):
    items = {it["qid"]: it for it in load_items()}
    plan = []
    for path in sorted(glob.glob("results/app_fix_*.jsonl")):
        if path.endswith("_judged.jsonl"):
            continue
        rows = load_jsonl(path)
        w = JsonlAppender(path.replace(".jsonl", "_judged.jsonl"), key=lambda r: (r["qid"], r["cond"]))
        todo = [r for r in rows if (r["qid"], r["cond"]) not in w.done]
        print(os.path.basename(path), "judged", len(rows) - len(todo), "todo", len(todo), flush=True)
        plan.append((path, w, todo))
    model = None
    if any(t for _, _, t in plan):
        tok, model = load_reader("Qwen3-14B")
    for path, w, todo in plan:
        for r in tqdm(todo, desc=os.path.basename(path)):
            if not r["skipped"]:
                it = items[r["qid"]]
                text = JUDGE.format(d0=it["dates"][0], d1=it["dates"][1], old=it["old_ev"] or "(not available)",
                                    new=it["new_ev"] or "(not available)", q=it["question"], a=it["answer"], r=r["response"])
                lp = letter_logprobs(tok, model, text)
                r["label"] = "ABC"[max(range(3), key=lambda i: lp[i])]
                r["lp"] = lp
            w.write(r)
        w.close()
        df = pd.DataFrame(w.rows)
        df = df[~df.skipped] if "skipped" in df else df
        if df.empty:
            continue
        df["correct"], df["stale"] = df.label.eq("A"), df.label.eq("B")
        print(os.path.basename(path), "\n", (df.groupby("cond")[["correct", "stale"]].mean() * 100).round(1).assign(
            n=df.groupby("cond").size()).to_string(), flush=True)
    if model is not None:
        del model
        free_gpu()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["read", "think", "judge", "check"])
    ap.add_argument("--models", default="Qwen3-4B,Qwen3-14B,Phi-4-mini,OLMo-2-13B-Instruct")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--budget", type=int, default=2048)
    args = ap.parse_args()
    if args.stage == "check":                     # structural checks only; prints no conversation text
        items = load_items()
        it = next(i for i in items if i["has_ev"])
        for c in conds_for(it) + ["S_rev_think", "T_oldlast_think"]:
            p = build(it, c)
            print(f"{c:22s} chars={len(p):6d} q_first={p.startswith('The user asks')} "
                  f"instr={INSTR[:20] in p} timeline={TIMELINE[:20] in p} dates={p.count(it['dates'][0][:10])}")
        a, b = build(it, "S_chrono_instr"), build(it, "S_rev_instr")
        print("chrono/rev same lines:", sorted(a.splitlines()) == sorted(b.splitlines()), "| differ:", a != b)
        print("final_answer timeline:", final_answer("S_rev_timeline", "1. x (2023)\n2. y (2024)\nAnswer: You have 4 bikes."))
    else:
        dict(read=read, think=think, judge=judge)[args.stage](args)


if __name__ == "__main__":
    main()
