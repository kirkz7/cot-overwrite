"""Application 4: the model's own long thinking (Qwen3-4B thinking mode, MATH-500 integer-answer problems).

Does the model's OWN thinking contain overwritten answer states, and how is the final answer read out?
  gen    sample thinking (T=0.6, top-p 0.95, top-k 20; budget 4096 tokens, stop at </think>)
  final  natural final answer after </think> (budget-exceeded traces get Qwen's early-exit sentence first)
  probe  readout at candidate boundaries: cut the thinking after each paragraph that states a candidate,
         force "The final answer is \\boxed{" and record which candidate comes out (self and Qwen3-14B as readers)
  stats  prevalence of revisions, final == last/first/most-supported candidate, abandoned-correct rate
usage: python run_app_thinking.py gen | final | probe --readers Qwen3-4B,Qwen3-14B | stats
"""
import argparse
import collections
import json
import os
import re
import time

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from app_common import tag, JsonlAppender, free_gpu, load_jsonl, load_reader
from early_answer import SUFFIX, answer_from_cache, parse, segments
from fastgen import GraphGen, auto_batch
from probe import PREFILL

GEN = "results/app_think_gen.jsonl"
FIN = "results/app_think_final.jsonl"
INSTR = "\nPlease reason step by step, and put your final answer within \\boxed{}."
EARLY_EXIT = "\n\nConsidering the limited time by the user, I have to give the solution based on the thinking directly now."
# an answer statement: "answer/result ... is|=|be|equals|: N" or \boxed{N}; hypotheticals ending in "?" are dropped
CAND = re.compile(r"(?i)(?:\b(?:answer|result)\b[^.\n?]{0,40}?(?:\bis\b|=|\bbe\b|\bequals\b|:)\s*\**\s*\$?\s*(?:\\boxed\{)?\s*"
                  r"|\\boxed\{\s*)(-?\d[\d,]*)(?![\d,]*\.\d)(?!\d)")


def problems():
    from huggingface_hub import hf_hub_download
    p = hf_hub_download("HuggingFaceH4/MATH-500", "test.jsonl", repo_type="dataset")
    rows = [json.loads(l) for l in open(p, encoding="utf-8")]
    out = []
    for i, r in enumerate(rows):
        a = r["answer"].replace(",", "").replace("\\!", "").strip()
        if re.fullmatch(r"-?\d+", a):
            out.append(dict(idx=i, problem=r["problem"], gold=int(a), level=r["level"], subject=r["subject"]))
    return out


def cands(text):
    """[(paragraph index, value)] for each answer statement, in order."""
    res = []
    for j, para in enumerate(text.split("\n\n")):
        for m in CAND.finditer(para):
            rest = para[m.end():m.end() + 3].lstrip("$*} ")
            if rest.startswith("?"):
                continue
            res.append((j, int(m.group(1).replace(",", ""))))
    return res


def head(tok, problem):
    return tok.apply_chat_template([{"role": "user", "content": problem + INSTR}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True).split("<think>")[0] + "<think>\n"


def gen(args):
    probs = problems()[:args.n]
    w = JsonlAppender(GEN, key=lambda r: r["idx"])
    todo = [p for p in probs if p["idx"] not in w.done]
    print("integer-answer problems", len(probs), "done", len(probs) - len(todo), "todo", len(todo), flush=True)
    if todo:
        tok, model = load_reader("Qwen3-4B")
        think_end = tok.convert_tokens_to_ids("</think>")
        stops = {think_end, tok.convert_tokens_to_ids("<|im_end|>")}
        enc = {p["idx"]: tok(head(tok, p["problem"]), add_special_tokens=False).input_ids for p in todo}
        L = max(map(len, enc.values())) + args.budget + 8
        bs = auto_batch(model, L, cap=args.bs)
        print("batch size", bs, "cache length", L, flush=True)
        gg = GraphGen(model, bs, L)
        t0, ntok = time.time(), 0
        for b in range(0, len(todo), bs):
            batch = todo[b:b + bs]
            g = torch.Generator(device="cuda").manual_seed(args.seed * 1_000_003 + batch[0]["idx"])   # per-batch seed
            out = gg.generate([enc[p["idx"]] for p in batch], args.budget, stops, tok.pad_token_id, sample=True, generator=g)
            recs = []
            for p, (row, stop) in zip(batch, out):
                ntok += len(row)
                recs.append(dict(p, thinking=tok.decode(row, skip_special_tokens=True).strip(),
                                 finished=stop == think_end, stop=stop, n_tokens=len(row)))
            w.write(*recs)
            print(f"{b + len(batch)}/{len(todo)}  {ntok / max(time.time() - t0, 1e-3):.0f} tok/s", flush=True)
    w.close()


def final(args):
    rows = load_jsonl(GEN)
    w = JsonlAppender(FIN, key=lambda r: r["idx"])
    todo = [r for r in rows if r["idx"] not in w.done]
    print("traces", len(rows), "done", len(rows) - len(todo), "todo", len(todo), flush=True)
    if todo:
        tok, model = load_reader("Qwen3-4B")
        im_end = tok.convert_tokens_to_ids("<|im_end|>")
        enc = [tok(head(tok, r["problem"]) + r["thinking"] + ("" if r["finished"] else EARLY_EXIT) + "\n</think>\n\n",
                   add_special_tokens=False).input_ids for r in todo]
        L = max(map(len, enc)) + args.answer_budget + 8
        bs = auto_batch(model, L, cap=args.bs)
        print("batch size", bs, "cache length", L, flush=True)
        gg = GraphGen(model, bs, L)
        for b in tqdm(range(0, len(todo), bs)):
            out = gg.generate(enc[b:b + bs], args.answer_budget, {im_end}, tok.pad_token_id, sample=False)
            recs = []
            for r, (row, _) in zip(todo[b:b + bs], out):
                txt = tok.decode(row, skip_special_tokens=True)
                boxes = re.findall(r"\\boxed\{([^{}]*)\}", txt)
                m = re.search(r"-?\d+", boxes[-1].replace(",", "")) if boxes else None
                recs.append(dict(idx=r["idx"], answer_text=txt[-300:], final=int(m.group()) if m else None))
            w.write(*recs)
    w.close()


@torch.no_grad()
def forced_answers(tok, model, r, points, suffix_ids, stop_ids):
    """{point: integer answer forced after the first `point` paragraphs}; one prefill, KV cache cropped per point."""
    paras = r["thinking"].split("\n\n")
    segs = segments(tok, r["problem"] + INSTR, paras)
    ids = [t for s in segs for t in s]
    ends = [sum(map(len, segs[:j + 1])) for j in range(len(segs))]
    with sdpa_kernel(PREFILL, set_priority=True):
        cache = model(torch.tensor([ids], device="cuda"), use_cache=True, logits_to_keep=1).past_key_values
    res = {}
    for p in sorted(points, reverse=True):
        cache.crop(ends[p])
        res[p] = parse(tok.decode(answer_from_cache(model, cache, suffix_ids, stop_ids), skip_special_tokens=True))
    del cache
    torch.cuda.empty_cache()
    return res


def complete_traces(rows):
    """Probe rows of a trace are written together; keep only traces whose rows are all present."""
    cnt = collections.Counter(r["idx"] for r in rows)
    return [r for r in rows if cnt[r["idx"]] == r["n_points"]]


@torch.no_grad()
def probe(args):
    rows = [r for r in load_jsonl(GEN) if len({v for _, v in cands(r["thinking"])}) >= 2]
    print("revision traces", len(rows), flush=True)
    for name in args.readers.split(","):
        w = JsonlAppender(f"results/app_think_probe_{tag(name)}.jsonl", key=lambda r: r["idx"], valid=complete_traces)
        todo = [r for r in rows if r["idx"] not in w.done]
        print(name, "done", len(rows) - len(todo), "todo", len(todo), flush=True)
        if todo:
            tok, model = load_reader(name)
            stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
            suffix_ids = tok(SUFFIX, add_special_tokens=False).input_ids
            for r in tqdm(todo, desc=name):
                cs = cands(r["thinking"])
                points = sorted({j + 1 for j, _ in cs})            # cut after paragraph j (segs index j+1)
                res = forced_answers(tok, model, r, points, suffix_ids, stop_ids)
                recs = []
                for p in points:
                    seen = [v for j, v in cs if j + 1 <= p]
                    recs.append(dict(reader=name, idx=r["idx"], gold=r["gold"], point=p, n_points=len(points),
                                     n_paras=len(r["thinking"].split("\n\n")), latest=seen[-1],
                                     earlier=sorted(set(seen[:-1]) - {seen[-1]}), forced=res[p]))
                w.write(*recs)
            del tok, model
            free_gpu()
        w.close()


def stats(args):
    import pandas as pd
    gen_rows = {r["idx"]: r for r in load_jsonl(GEN)}
    fin = {r["idx"]: r["final"] for r in load_jsonl(FIN)}
    recs = []
    for idx, r in gen_rows.items():
        vals = [v for _, v in cands(r["thinking"])]
        seq = [v for i, v in enumerate(vals) if i == 0 or v != vals[i - 1]]
        cnt = collections.Counter(vals)
        top = max(cnt.values()) if cnt else 0
        mode = [v for v in reversed(vals) if cnt[v] == top][0] if vals else None
        fa = fin.get(idx)
        recs.append(dict(idx=idx, level=r["level"], finished=r["finished"], n_tokens=r["n_tokens"], n_stmt=len(vals),
                         n_distinct=len(cnt), n_switch=len(seq) - 1, final=fa, correct=fa == r["gold"],
                         last_ok=bool(vals) and vals[-1] == r["gold"], first_ok=bool(vals) and vals[0] == r["gold"],
                         mode_ok=mode == r["gold"], final_is_last=bool(vals) and fa == vals[-1],
                         final_is_earlier=bool(vals) and fa in vals[:-1] and fa != vals[-1],
                         abandoned_correct=r["gold"] in vals[:-1] and fa != r["gold"],
                         rescued=bool(vals) and vals[0] != r["gold"] and fa == r["gold"]))
    df = pd.DataFrame(recs)
    print("final answer parsed", f"{df.final.notna().mean():.1%}")
    df = df[df.final.notna()]
    print("traces", len(df), "| finished thinking", f"{df.finished.mean():.1%}", "| median tokens", int(df.n_tokens.median()),
          "| final accuracy", f"{df.correct.mean():.1%}")
    print("traces with >=1 answer statement", f"{(df.n_stmt > 0).mean():.1%}", "| with >=2 distinct candidates (revision)",
          f"{(df.n_distinct >= 2).mean():.1%}", "| mean switches among revision traces", round(df[df.n_distinct >= 2].n_switch.mean(), 2))
    rv = df[df.n_distinct >= 2]
    print("revision traces n", len(rv), "| final == last candidate", f"{rv.final_is_last.mean():.1%}",
          "| final == superseded candidate", f"{rv.final_is_earlier.mean():.1%}")
    print("readout accuracy on revision traces: last", f"{rv.last_ok.mean():.1%}", "first", f"{rv.first_ok.mean():.1%}",
          "most-stated", f"{rv.mode_ok.mean():.1%}", "actual final", f"{rv.correct.mean():.1%}")
    print("abandoned a correct candidate (all traces)", f"{df.abandoned_correct.mean():.1%}", "| among revision traces",
          f"{rv.abandoned_correct.mean():.1%}", "| rescued (first candidate wrong, final right)", f"{rv.rescued.mean():.1%}")
    lv = df.groupby("level").agg(rev=("n_distinct", lambda s: (s >= 2).mean()), acc=("correct", "mean"), n=("idx", "size"))
    print("by level (revision rate, accuracy):\n", lv.round(3).to_string())
    pct = lambda x: round(float(x) * 100, 1)
    summary = dict(n=len(df), finished=pct(df.finished.mean()), median_tokens=int(df.n_tokens.median()), accuracy=pct(df.correct.mean()),
                   any_statement=pct((df.n_stmt > 0).mean()), revision=pct((df.n_distinct >= 2).mean()), n_revision=len(rv),
                   final_is_last=pct(rv.final_is_last.mean()), final_is_superseded=pct(rv.final_is_earlier.mean()),
                   readout_acc=dict(last=pct(rv.last_ok.mean()), first=pct(rv.first_ok.mean()), most_stated=pct(rv.mode_ok.mean()),
                                    actual=pct(rv.correct.mean())),
                   abandoned_correct_all=pct(df.abandoned_correct.mean()), abandoned_correct_rev=pct(rv.abandoned_correct.mean()),
                   rescued_rev=pct(rv.rescued.mean()),
                   by_level={int(k): dict(rev=pct(r.rev), acc=pct(r.acc), n=int(r.n)) for k, r in lv.iterrows()}, probe={})
    for p in sorted(os.listdir("results")):
        if p.startswith("app_think_probe_") and p.endswith(".jsonl"):
            pr = pd.DataFrame(complete_traces(load_jsonl("results/" + p)))
            if pr.empty:
                continue
            pr["is_latest"] = pr.forced == pr.latest
            pr["is_earlier"] = [f in e for f, e in zip(pr.forced, pr.earlier)]
            pr["has_earlier"] = pr.earlier.map(len) > 0
            sub = pr[pr.has_earlier]
            print(p, "| cut points", len(pr), "| with a superseded candidate before the cut", len(sub),
                  "| forced == latest", f"{sub.is_latest.mean():.1%}", "| forced == superseded", f"{sub.is_earlier.mean():.1%}")
            summary["probe"][p[len("app_think_probe_"):-len(".jsonl")]] = dict(
                cuts=len(pr), cuts_with_superseded=len(sub), latest=pct(sub.is_latest.mean()), superseded=pct(sub.is_earlier.mean()))
    json.dump(summary, open("results/app_think_summary.json", "w", encoding="utf-8"), indent=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "final", "probe", "stats", "check"])
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--bs", type=int, default=8)
    ap.add_argument("--budget", type=int, default=6144)
    ap.add_argument("--answer_budget", type=int, default=1280)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--readers", default="Qwen3-4B,Qwen3-14B")
    args = ap.parse_args()
    if args.stage == "check":
        ps = problems()
        print("integer-answer problems", len(ps), "levels", collections.Counter(p["level"] for p in ps))
        demo = ("Let me compute. So the answer is 12.\n\nWait, is the answer 15? Let me check.\n\nNo, the result is 1,234."
                "\n\nTherefore the final answer is \\boxed{7}.\n\nThe answer should be 3.5 or 2.")
        print(cands(demo))
    else:
        dict(gen=gen, final=final, probe=probe, stats=stats)[args.stage](args)


if __name__ == "__main__":
    main()
