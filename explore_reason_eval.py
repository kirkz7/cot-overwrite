"""Exploration E12 evaluation: does a LoRA trained to write time-organized reasoning (gen_reason_data.py) transfer to
formats it never saw? Readers: base Qwen3-4B, the answer-only decoupled LoRA (q4-dec-s0), E12 decoupled, E12 control.
  convo  ConvoMem changing-evidence, personas 50-99 (held out: never used in any experiment before E12), two dated
         conversations, chronological / newest-first                                          -> judged by Qwen3-14B
  lme    LongMemEval knowledge-update sessions, chrono / newest-first (seen before E12: reported, but flagged)  -> judged
  email  our synthetic email threads with 10 / 40 / 200 entries, oldest_first / newest_first   -> exact number
The prompts are the tasks' own (no added instruction): a model may answer directly or reason first. The scored answer is
the text after the last "Answer:" if present, else the first line. Greedy, up to 320 new tokens.
usage: python explore_reason_eval.py run --models ... | judge | stats
"""
import argparse
import os

import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, out_tag, strip_think, to_think, JsonlAppender, chat_prompt, first_number, free_gpu, load_jsonl, load_reader
import run_app_logs as logs
import run_app_memory as mem
from explore_route import convo_items
from explore_think import log_items
from run_app_fix import generate, generate_think

OUT = "results/explore_reason_{}.jsonl"


def extract(text):
    text = strip_think(text)                                          # E18: the visible reply only
    return text.rsplit("Answer:", 1)[1].strip().split("\n")[0] if "Answer:" in text else text.strip().split("\n")[0]


def items(tok):
    out = []
    for it in convo_items(tok, range(50, 100), per=4):
        out.append(dict(task="convo", id=it["id"], cond=it["cond"], prompt=it["prompt"]))
    for it in mem.load_items():
        for cond in ("S_chrono_dated", "S_rev_dated"):
            out.append(dict(task="lme", id=it["qid"], cond=cond, prompt=chat_prompt(tok, mem.build(it, cond))))
    for n_ent in (10, 40, 200):
        for it in log_items(20, n_ent):
            for order in ("oldest_first", "newest_first"):
                out.append(dict(task=f"email{n_ent}", id=str(it["seed"]), cond=order,
                                prompt=chat_prompt(tok, logs.render(it, "email", order)[0]), hist=it["history"]))
    return out


@torch.no_grad()
def run(args):
    for name in args.models.split(","):
        w = JsonlAppender(OUT.format(out_tag(name, args.think)), key=lambda r: (r["task"], r["id"], r["cond"]))
        tok, model = load_reader(name)
        todo = [x for x in items(tok) if (x["task"], x["id"], x["cond"]) not in w.done
                and (not args.tasks or x["task"] in args.tasks.split(","))]
        print(name, "todo", len(todo), flush=True)
        for x in tqdm(todo, desc=name):
            if args.think:                                            # Qwen3 recommended sampling (10-06)
                text = generate_think(tok, model, to_think(x["prompt"]), 1024, (x["task"], x["id"], x["cond"]))
            else:
                text = generate(tok, model, x["prompt"], 320)
            rec = dict(task=x["task"], id=x["id"], cond=x["cond"], full=text, response=extract(text),
                       reasoned="Answer:" in text)
            if "hist" in x:
                pred = first_number(rec["response"])
                rec.update(correct=pred == x["hist"][-1], stale=pred in x["hist"][:-1])
            w.write(rec)
            if len(x["prompt"]) > 12000:
                torch.cuda.empty_cache()
        w.close()
        del tok, model
        free_gpu()


@torch.no_grad()
def judge(args):
    import glob
    lme = {it["qid"]: it for it in mem.load_items()}
    tok = model = None
    for path in sorted(glob.glob(OUT.format("*"))):
        if path.endswith("_judged.jsonl"):
            continue
        rows = [r for r in load_jsonl(path) if r["task"] in ("convo", "lme")]
        w = JsonlAppender(path.replace(".jsonl", "_judged.jsonl"), key=lambda r: (r["task"], r["id"], r["cond"]))
        todo = [r for r in rows if (r["task"], r["id"], r["cond"]) not in w.done]
        print(os.path.basename(path), "judge todo", len(todo), flush=True)
        if todo and model is None:
            tok, model = load_reader("Qwen3-14B")
            conv = {it["id"]: it for it in convo_items(tok, range(50, 100), per=4)}
        for r in tqdm(todo):
            if r["task"] == "lme":
                it = lme[r["id"]]
                text = mem.JUDGE.format(d0=it["dates"][0], d1=it["dates"][1], old=it["old_ev"] or "(not available)",
                                        new=it["new_ev"] or "(not available)", q=it["question"], a=it["answer"], r=r["response"])
            else:
                it = conv[r["id"]]
                text = mem.JUDGE.format(d0=it["dates"][0], d1=it["dates"][1], old=it["ev"][0], new=it["ev"][1],
                                        q=it["q"], a=it["ref"], r=r["response"])
            lp = mem.letter_logprobs(tok, model, text)
            lab = "ABC"[max(range(3), key=lambda i: lp[i])]
            w.write(dict(r, label=lab, correct=lab == "A", stale=lab == "B"))
        w.close()
    if model is not None:
        del tok, model
        free_gpu()


def stats(args):
    import glob
    from analyze_lora import boot
    frames = {}
    for path in sorted(glob.glob(OUT.format("*"))):
        if path.endswith("_judged.jsonl"):
            continue
        name = os.path.basename(path)[len("explore_reason_"):-len(".jsonl")]
        a = pd.DataFrame([r for r in load_jsonl(path) if r["task"].startswith("email")])
        jp = path.replace(".jsonl", "_judged.jsonl")
        b = pd.DataFrame(load_jsonl(jp)) if os.path.exists(jp) else pd.DataFrame()
        frames[name] = pd.concat([a, b], ignore_index=True)
    rows = []
    for name, df in frames.items():
        for (task, cond), g in df.groupby(["task", "cond"]):
            rows.append(dict(model=name, task=task, cond=cond, n=len(g), correct=g.correct.mean() * 100,
                             stale=g.stale.mean() * 100, reasoned=g.reasoned.mean() * 100))
    t = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(t.pivot_table(index=["task", "cond"], columns="model", values="correct").round(1).to_string())
    print("\nreasoned (%):\n", t.pivot_table(index=["task", "cond"], columns="model", values="reasoned").round(0).to_string())
    dec = next((k for k in frames if "e12-dec" in k), None)
    for other in [k for k in frames if k != dec]:
        if dec is None:
            break
        out = []
        for (task, cond), g in frames[dec].groupby(["task", "cond"]):
            h = frames[other][(frames[other].task == task) & (frames[other].cond == cond)]
            m = g.merge(h, on="id", suffixes=("_a", "_b"))
            if len(m):
                d, lo, hi, _ = boot(m.correct_a.astype(float), m.correct_b.astype(float))
                out.append(dict(task=task, cond=cond, n=len(m), diff=d, lo=lo, hi=hi))
        print(f"\n{dec} - {other} (correct %, paired bootstrap 95% CI)\n", pd.DataFrame(out).round(1).to_string(index=False))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "judge", "stats", "check"])
    ap.add_argument("--models", default="Qwen3-4B")
    ap.add_argument("--tasks", default="", help="comma list of convo,lme,email<n> (default: all)")
    ap.add_argument("--think", action="store_true", help="Qwen3 thinking mode (E18); results go to <tag>+think")
    args = ap.parse_args()
    if args.stage == "check":
        from transformers import AutoTokenizer
        import collections
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        xs = items(tok)
        print(collections.Counter((x["task"], x["cond"]) for x in xs))
    else:
        {"run": run, "judge": judge, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()
