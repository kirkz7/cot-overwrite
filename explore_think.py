"""Exploration E1c (EXPLORE_PLAN.md): what does thinking mode do with a misleading presentation order?

Qwen3-4B thinking mode (sampled: T 0.6, top-p 0.95, top-k 20; 2048-token budget, then the official early-exit
sentence), with the thinking text saved this time:
  memory  LongMemEval knowledge-update items, two dated sessions, chronological vs newest-first (78 items each)
  logs    our synthetic email threads (run_app_logs.py items), oldest_first vs newest_first (20 items per k = 1, 2, 4)
Questions: (1) does the thinking restate the records in their presentation order (which date / value is mentioned
first)? (2) what does the thinking itself conclude, and does that conclusion go stale more often in the misleading
order? (3) does the final answer follow the thinking's conclusion? (4) when the thinking mentions the up-to-date value,
how often is the answer still stale?
Exploratory; the threshold is written in EXPLORE_PLAN.md before running. LongMemEval text is never printed; the
`show` stage prints one trace of our own synthetic email task.
usage: python explore_think.py gen [--models Qwen3-4B]
       python explore_think.py judge
       python explore_think.py stats
       python explore_think.py show
"""
import argparse
import glob
import os
import re

import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, JsonlAppender, first_number, free_gpu, load_jsonl, load_reader
import run_app_logs as logs
from run_app_fix import EARLY_EXIT, build as fix_build
from run_app_memory import JUDGE, letter_logprobs, load_items

BUDGET = 2048
TAIL = 800          # characters at the end of the thinking that the judge reads as its conclusion
MONTHS = ["January", "February", "March", "April", "May", "June", "July", "August", "September", "October",
          "November", "December"]


PADS = [40, 200]       # length control (added 2026-10-03): email threads with 40 / 200 entries (~1.7k / ~8.4k tokens)


def log_items(n, n_entries=10):
    """Email items; n_entries > 10 pads the thread with more unrelated emails (same targets and seeds)."""
    old = logs.N_ENTRIES
    logs.N_ENTRIES = n_entries
    try:
        return [logs.make_item(k, 7_000_000 + k * 10_000 + i) for k in logs.KS for i in range(n)]
    finally:
        logs.N_ENTRIES = old


def task_entries(task):
    return 10 if task == "logs" else int(task[4:])


def jobs(n_logs, pad=False):
    out = []
    for it in load_items():
        for cond in ("S_chrono_think", "S_rev_think"):
            out.append(("memory", it["qid"], cond, fix_build(it, cond)))
    for task in ["logs"] + ([f"logs{n}" for n in PADS] if pad else []):
        for it in log_items(n_logs, task_entries(task)):
            for order in ("oldest_first", "newest_first"):
                out.append((task, str(it["seed"]), order, logs.render(it, "email", order)[0]))
    return out


@torch.no_grad()
def gen(args):
    from fastgen import GraphGen, auto_batch
    for name in args.models.split(","):
        w = JsonlAppender(f"results/explore_think_{tag(name)}.jsonl", key=lambda r: (r["task"], r["id"], r["cond"]))
        todo_all = [j for j in jobs(args.n_logs, args.pad) if (j[0], j[1], j[2]) not in w.done]
        print(name, "todo", len(todo_all), flush=True)
        if not todo_all:
            w.close()
            continue
        tok, model = load_reader(name)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")

        def head(text):
            return tok.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=True).split("<think>")[0] + "<think>\n"
        for task in sorted({j[0] for j in todo_all}, key=lambda t: t != "logs"):   # one static cache per task
            todo = sorted([j for j in todo_all if j[0] == task], key=lambda j: len(j[3]))
            if not todo:
                continue
            enc = [tok(head(j[3]), add_special_tokens=False).input_ids for j in todo]
            L = max(map(len, enc)) + BUDGET + 256
            bs = auto_batch(model, L, cap=8)
            print(task, "batch size", bs, "cache length", L, flush=True)
            gg = GraphGen(model, bs, L)
            for b in tqdm(range(0, len(todo), bs), desc=task):
                g = torch.Generator(device="cuda").manual_seed(5000 + b)
                out = gg.generate(enc[b:b + bs], BUDGET, {im_end}, tok.pad_token_id, sample=True, generator=g)
                recs = []
                for (task_, iid, cond, _), ids, (row, stop) in zip(todo[b:b + bs], enc[b:b + bs], out):
                    text = tok.decode(row, skip_special_tokens=False)
                    if "</think>" in text:
                        thought, ans = text.split("</think>", 1)
                        ans, finished = ans.replace("<|im_end|>", "").strip(), True
                    else:                                       # budget hit: official early exit, then a short answer
                        thought, finished, ans = text, False, ""
                        p2 = tok.decode(ids, skip_special_tokens=False) + text + EARLY_EXIT + "\n</think>\n\n"
                        for x in gg.generate([tok(p2, add_special_tokens=False).input_ids], 96, {im_end},
                                             tok.pad_token_id, sample=False):
                            ans = tok.decode(x[0], skip_special_tokens=True).strip()
                    recs.append(dict(model=name, task=task_, id=iid, cond=cond, finished=finished,
                                     n_think=len(tok(thought, add_special_tokens=False).input_ids),
                                     thought=thought.strip(), answer=ans, response=ans.split("\n")[0][:400]))
                w.write(*recs)
            del gg
            torch.cuda.empty_cache()
        del tok, model
        free_gpu()
        w.close()


@torch.no_grad()
def swap(args):
    """E1c swap (added 2026-10-03): does the answer follow the thinking or the input order? The saved thinking of one
    order is put after the input of the other order (and, as a control, after its own input), then the answer is
    produced greedily. Conds: swap_rev_chronothought, swap_chrono_revthought, own_rev, own_chrono."""
    from fastgen import GraphGen, auto_batch
    items = {it["qid"]: it for it in load_items()}
    for name in args.models.split(","):
        path = f"results/explore_think_{tag(name)}.jsonl"
        th = {(r["id"], r["cond"]): r["thought"] for r in load_jsonl(path) if r["task"] == "memory"}
        plan = [("swap_rev_chronothought", "S_rev_think", "S_chrono_think"),
                ("swap_chrono_revthought", "S_chrono_think", "S_rev_think"),
                ("own_rev", "S_rev_think", "S_rev_think"), ("own_chrono", "S_chrono_think", "S_chrono_think")]
        w = JsonlAppender(path, key=lambda r: (r["task"], r["id"], r["cond"]))
        todo = [(q, c, inp, src) for q in items for c, inp, src in plan
                if ("memory", q, c) not in w.done and (q, src) in th]
        print(name, "swap todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")

        def prompt(q, inp, src):
            head = tok.apply_chat_template([{"role": "user", "content": fix_build(items[q], inp)}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=True).split("<think>")[0]
            return head + "<think>\n" + th[(q, src)] + "\n</think>\n\n"
        enc = [tok(prompt(q, inp, src), add_special_tokens=False).input_ids for q, c, inp, src in todo]
        L = max(map(len, enc)) + 104
        gg = GraphGen(model, auto_batch(model, L, cap=8), L)
        for b in tqdm(range(0, len(todo), gg.B), desc=f"{name} swap"):
            out = gg.generate(enc[b:b + gg.B], 96, {im_end}, tok.pad_token_id, sample=False)
            recs = []
            for (q, c, inp, src), (row, stop) in zip(todo[b:b + gg.B], out):
                ans = tok.decode(row, skip_special_tokens=True).strip()
                recs.append(dict(model=name, task="memory", id=q, cond=c, finished=True, n_think=0,
                                 thought=th[(q, src)], answer=ans, response=ans.split("\n")[0][:400]))
            w.write(*recs)
        del tok, model, gg
        free_gpu()
        w.close()


@torch.no_grad()
def notes(args):
    """E11 (added 2026-10-03): format-agnostic two-step harness. Step 1 is the saved thinking (E1c). Step 2 removes the
    original records and asks the question again with only the model's own notes, so the input's last-presented value
    cannot pull the answer. No parser, no format-specific component. Conds: notes_chrono, notes_rev."""
    from fastgen import GraphGen, auto_batch
    items = {it["qid"]: it for it in load_items()}
    for name in args.models.split(","):
        path = f"results/explore_think_{tag(name)}.jsonl"
        th = {(r["id"], r["cond"]): r["thought"] for r in load_jsonl(path) if r["task"] == "memory"}
        w = JsonlAppender(path, key=lambda r: (r["task"], r["id"], r["cond"]))
        plan = [("notes_chrono", "S_chrono_think"), ("notes_rev", "S_rev_think")]
        todo = [(q, c, src) for q in items for c, src in plan if ("memory", q, c) not in w.done and (q, src) in th]
        print(name, "notes todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        im_end = tok.convert_tokens_to_ids("<|im_end|>")

        def prompt(q, src):
            user = ("Below are your own notes, written while reading the user's past chat sessions (the sessions themselves "
                    "are not shown again).\n\nNotes:\n" + th[(q, src)].strip() +
                    f"\n\nCurrent date: {items[q]['qdate']}\nBased on your notes, answer the user's question in one short "
                    f"sentence.\nQuestion: {items[q]['question']}")
            return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                           add_generation_prompt=True, enable_thinking=False)
        enc = [tok(prompt(q, src), add_special_tokens=False).input_ids for q, c, src in todo]
        L = max(map(len, enc)) + 104
        gg = GraphGen(model, auto_batch(model, L, cap=8), L)
        for b in tqdm(range(0, len(todo), gg.B), desc=f"{name} notes"):
            out = gg.generate(enc[b:b + gg.B], 96, {im_end}, tok.pad_token_id, sample=False)
            recs = []
            for (q, c, src), (row, stop) in zip(todo[b:b + gg.B], out):
                ans = tok.decode(row, skip_special_tokens=True).strip()
                recs.append(dict(model=name, task="memory", id=q, cond=c, finished=True, n_think=0,
                                 thought=th[(q, src)], answer=ans, response=ans.split("\n")[0][:400]))
            w.write(*recs)
        del tok, model, gg
        free_gpu()
        w.close()


@torch.no_grad()
def judge(args):
    """Memory rows: label the final answer and the end of the thinking (A up to date / B outdated / C neither)."""
    items = {it["qid"]: it for it in load_items()}
    plan = []
    for path in sorted(glob.glob("results/explore_think_*.jsonl")):
        if path.endswith("_judged.jsonl"):
            continue
        rows = [r for r in load_jsonl(path) if r["task"] == "memory"]
        w = JsonlAppender(path.replace(".jsonl", "_judged.jsonl"), key=lambda r: (r["id"], r["cond"]))
        plan.append((w, [r for r in rows if (r["id"], r["cond"]) not in w.done]))
    tok = model = None
    if any(t for _, t in plan):
        tok, model = load_reader("Qwen3-14B")
    for w, todo in plan:
        for r in tqdm(todo):
            it = items[r["id"]]
            labels = {}
            for key, resp in (("final", r["response"]), ("think", r["thought"][-TAIL:])):
                text = JUDGE.format(d0=it["dates"][0], d1=it["dates"][1], old=it["old_ev"] or "(not available)",
                                    new=it["new_ev"] or "(not available)", q=it["question"], a=it["answer"], r=resp)
                lp = letter_logprobs(tok, model, text)
                labels[key] = "ABC"[max(range(3), key=lambda i: lp[i])]
            w.write(dict(r, label_final=labels["final"], label_think=labels["think"]))
        w.close()
    if model is not None:
        del tok, model
        free_gpu()


def date_variants(d):
    """'2023/05/20 (Sat) 02:21' -> strings a thinking trace may use for that date."""
    m = re.match(r"(\d{4})/(\d\d)/(\d\d)", d)
    y, mo, da = m.groups()
    name = MONTHS[int(mo) - 1]
    return [f"{y}/{mo}/{da}", f"{y}-{mo}-{da}", f"{name} {int(da)}", f"{name[:3]} {int(da)}", f"{int(da)} {name}"]


def first_pos(text, variants):
    ps = [text.find(v) for v in variants if v in text]
    return min(ps) if ps else None


def words(s):
    return re.sub(r"[^a-z0-9 ]", " ", str(s).lower()).split()


def stats(args):
    items = {it["qid"]: it for it in load_items()}
    for path in sorted(glob.glob("results/explore_think_*_judged.jsonl")):
        df = pd.DataFrame(load_jsonl(path))
        name = os.path.basename(path)
        recs = []
        for r in df.itertuples():
            it = items[r.id]
            t = r.thought
            p_old, p_new = first_pos(t, date_variants(it["dates"][0])), first_pos(t, date_variants(it["dates"][1]))
            gold = words(it["answer"])
            tw = words(t)
            recs.append(dict(cond=r.cond, finished=r.finished, n_think=r.n_think,
                             final_stale=r.label_final == "B", final_ok=r.label_final == "A",
                             think_stale=r.label_think == "B", think_ok=r.label_think == "A",
                             agree=r.label_final == r.label_think,
                             both_dates=p_old is not None and p_new is not None,
                             new_date_first=(p_new < p_old) if p_old is not None and p_new is not None else None,
                             gold_in_think=bool(gold) and all(x in tw for x in gold),
                             waits=len(re.findall(r"\b(wait|but|however|actually|hmm)\b", t.lower()))))
        d = pd.DataFrame(recs)
        print(name)
        g = d.groupby("cond").agg(n=("final_ok", "size"), finished=("finished", "mean"), think_tokens=("n_think", "median"),
                                  final_ok=("final_ok", "mean"), final_stale=("final_stale", "mean"),
                                  think_ok=("think_ok", "mean"), think_stale=("think_stale", "mean"),
                                  final_follows_think=("agree", "mean"), both_dates=("both_dates", "mean"),
                                  waits=("waits", "mean"))
        pct = ["finished", "final_ok", "final_stale", "think_ok", "think_stale", "final_follows_think", "both_dates"]
        g[pct] = g[pct] * 100
        print(g.round(1).to_string())
        b = d[d.both_dates]
        print("new date mentioned before old date (traces naming both dates):",
              (b.groupby("cond").new_date_first.mean() * 100).round(1).to_dict(), "n", b.groupby("cond").size().to_dict())
        gi = d[d.gold_in_think]
        print("final stale when the thinking contains the up-to-date value:",
              (gi.groupby("cond").final_stale.mean() * 100).round(1).to_dict(), "n", gi.groupby("cond").size().to_dict())
        print("final stale when the thinking's own conclusion is up to date:",
              (d[d.think_ok].groupby("cond").final_stale.mean() * 100).round(1).to_dict(),
              "n", d[d.think_ok].groupby("cond").size().to_dict())
    for path in sorted(glob.glob("results/explore_think_*.jsonl")):
        if path.endswith("_judged.jsonl"):
            continue
        rows = [r for r in load_jsonl(path) if r["task"].startswith("logs")]
        if not rows:
            continue
        its = {(t, str(it["seed"])): it for t in {r["task"] for r in rows}
               for it in log_items(args.n_logs, task_entries(t))}
        recs = []
        for r in rows:
            h = its[(r["task"], r["id"])]["history"]
            pred = first_number(r["response"])
            # target values in the order the thinking mentions them
            seq = [int(x) for x in re.findall(r"\b\d+\b", r["thought"]) if int(x) in h]
            recs.append(dict(task=r["task"], order=r["cond"], correct=pred == h[-1], stale=pred in h[:-1],
                             think_last_is_latest=bool(seq) and seq[-1] == h[-1],
                             think_first_is_latest=bool(seq) and seq[0] == h[-1],
                             n_think=r["n_think"], finished=r["finished"]))
        d = pd.DataFrame(recs)
        print(os.path.basename(path), "logs (email)")
        g = ["task", "order"]
        print((d.groupby(g)[["correct", "stale", "think_last_is_latest", "think_first_is_latest", "finished"]]
               .mean() * 100).round(1).assign(think_tokens=d.groupby(g).n_think.median(),
                                               n=d.groupby(g).size()).to_string())


def show(args):
    """One full trace from our own synthetic email task (no LongMemEval text)."""
    path = f"results/explore_think_{tag(args.models.split(',')[0])}.jsonl"
    rows = [r for r in load_jsonl(path) if r["task"] == "logs" and r["cond"] == "newest_first"]
    its = {str(it["seed"]): it for it in log_items(args.n_logs)}
    r = rows[args.i]
    it = its[r["id"]]
    print("history (oldest -> newest):", it["history"], "| target:", it["target"])
    print("=" * 30, "prompt\n" + logs.render(it, "email", "newest_first")[0])
    print("=" * 30, "thinking\n" + r["thought"])
    print("=" * 30, "answer\n" + r["answer"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["gen", "judge", "stats", "show", "swap", "notes"])
    ap.add_argument("--models", default="Qwen3-4B")
    ap.add_argument("--n_logs", type=int, default=20, help="email items per k (k = 1, 2, 4)")
    ap.add_argument("--i", type=int, default=0, help="show: which trace")
    ap.add_argument("--pad", action="store_true", help="gen: also the padded email threads (length control)")
    args = ap.parse_args()
    {"gen": gen, "judge": judge, "stats": stats, "show": show, "swap": swap, "notes": notes}[args.stage](args)


if __name__ == "__main__":
    main()
