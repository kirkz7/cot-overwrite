"""Long-dialogue test sets beyond LongMemEval (10-04, step 1 of the plan agreed with the user): measure the order effect
on BASE models only, before any new training. These sets are frozen as held-out tests for any later fix.
  convo_long  ConvoMem changing-evidence, personas 50-99 (never used before E12; E12 used 6-message excerpts of them,
              this uses the FULL 40-message conversations), the two conversations dated by us (older < newer, as in the
              source) and shown chronological or newest-first; free-text answer judged by Qwen3-14B (A / B / C).
  personamem  PersonaMem-v1 32k, question types track_full_preference_evolution and
              recalling_the_reasons_behind_previous_updates; the context up to the question is split into its sessions
              (each starts with the persona system message), each session gets a date (increasing, source order), and
              the sessions are shown chronological or newest-first; 4-way multiple choice, letter answer.
Pre-registered (EXPLORE_PLAN.md, before running): the effect replicates on a set if Qwen3-4B's newest-first accuracy is
>= 10 points below chronological with a paired-bootstrap 95% CI excluding 0. No conversation text is printed.
usage: python explore_longconv.py run --models Qwen3-4B,Phi-4-mini | judge | stats | check
"""
import argparse
import glob
import json
import os
import random
import re

import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, out_tag, strip_think, JsonlAppender, chat_prompt, free_gpu, load_jsonl, load_reader
import run_app_memory as mem
from explore_probe import CONVO_GLOB, LONG_SDPA, sdpa_kernel
from run_app_fix import generate, generate_think

OUT = "results/longconv_{}.jsonl"
PM = glob.glob(r"D:\hf_cache\hub\datasets--bowen-upenn--PersonaMem-v1\snapshots\*")
QTYPES = ("track_full_preference_evolution", "recalling_the_reasons_behind_previous_updates")


def convo_long(personas=range(50, 100), per=4):
    files = sorted(glob.glob(CONVO_GLOB))
    out = []
    for pi in personas:
        d = json.load(open(files[pi], encoding="utf-8"))
        for k, ev in enumerate(d["evidence_items"][:per]):
            texts = [m["text"] for m in ev["message_evidences"]]
            convs = ev["conversations"]
            if not all(any(m["text"] == t for m in c["messages"]) for c, t in zip(convs, texts)):
                continue
            r = random.Random(f"long{pi}-{k}")
            dates = ["2024-%02d-%02d" % (r.randint(1, 5), r.randint(1, 28)), "2024-%02d-%02d" % (r.randint(7, 11), r.randint(1, 28))]
            for cond, order in (("chrono", [0, 1]), ("rev", [1, 0])):
                blocks = [f"### Conversation (date: {dates[o]})\n" + "\n".join(f"{m['speaker']}: {m['text']}" for m in convs[o]["messages"])
                          for o in order]
                user = ("Here are records of your past conversations with the user.\n\n" + "\n\n".join(blocks) +
                        f"\n\nCurrent date: 2024-12-01\nBased on the information above, answer the user's question in one short "
                        f"sentence.\nQuestion: {ev['question']}")
                out.append(dict(task="convo_long", id=f"{pi}-{k}", cond=cond, user=user, q=ev["question"], ref=ev["answer"],
                                ev=texts, dates=dates))
    return out


def personamem(qtypes=QTYPES, conds=("chrono", "rev")):
    # conds "nohist" (persona + question + options, no sessions) and "bare" (question + options only) are the
    # 10-04 PM-diag controls: does the question need the history at all?
    root = PM[0]
    q = pd.read_csv(os.path.join(root, "questions_32k.csv"))
    if qtypes is not None:
        q = q[q.question_type.isin(qtypes)]
    ctx = {}
    for line in open(os.path.join(root, "shared_contexts_32k.jsonl"), encoding="utf-8"):
        ctx.update(json.loads(line))
    out = []
    for r in q.itertuples():
        turns = ctx[r.shared_context_id][: int(r.end_index_in_shared_context) + 1]
        sessions, cur = [], []
        for t in turns:
            if t["role"] == "system" and cur:
                sessions.append(cur); cur = []
            cur.append(t)
        sessions.append(cur)
        persona = sessions[0][0]["content"] if sessions[0][0]["role"] == "system" else ""
        body = [[t for t in s if t["role"] != "system"] for s in sessions]
        body = [b for b in body if b]
        rnd = random.Random(str(r.question_id))
        start = rnd.randint(1, 300)
        dates = [(pd.Timestamp("2023-01-01") + pd.Timedelta(days=start + 45 * i + rnd.randint(0, 20))).strftime("%Y-%m-%d")
                 for i in range(len(body))]
        opts = eval(r.all_options) if isinstance(r.all_options, str) else list(r.all_options)
        tail = (f"The user now says: {r.user_question_or_message}\n\n"
                "Which response is best for this user now? Options:\n" + "\n".join(opts) +
                "\n\nAnswer with the letter of the best option only, e.g. (a).")
        for cond in conds:
            if cond == "bare":
                user = tail
            elif cond == "nohist":
                user = f"Persona of the user you have been talking with:\n{persona}\n\n" + tail
            else:
                idx = list(range(len(body))) if cond == "chrono" else list(range(len(body)))[::-1]
                blocks = [f"### Session (date: {dates[i]})\n" + "\n".join(t["content"] for t in body[i]) for i in idx]
                user = (f"Persona of the user you have been talking with:\n{persona}\n\nHere are your past chat sessions with "
                        f"this user.\n\n" + "\n\n".join(blocks) + "\n\n" + tail)
            out.append(dict(task="personamem", id=str(r.question_id), cond=cond, user=user, qtype=r.question_type,
                            gold=str(r.correct_answer).strip().strip("()").lower()[:1], n_sessions=len(body)))
    return out


def letter(text):
    m = re.search(r"\(?([a-d])\)", text.lower()) or re.search(r"\b([a-d])\b", text.lower())
    return m.group(1) if m else None


def pm_pred(full):
    full = strip_think(full)                                          # E18: ignore the thought
    # 10-04 fix: reasoning readers (E13) end with "The best answer is (d)." and no "Answer:", so the first line
    # (the list header) held no letter. Rule: text after the last "Answer:" if present; else the last "(x)" in the
    # whole output; else the first line. Identical for every model; base-model predictions are unchanged.
    if "Answer:" in full:
        return letter(full.rsplit("Answer:", 1)[1].strip().split("\n")[0])
    ms = re.findall(r"\(([a-d])\)", full.lower())
    return ms[-1] if ms else letter(full.strip().split("\n")[0])


@torch.no_grad()
def run(args):
    for name in args.models.split(","):
        w = JsonlAppender(OUT.format(out_tag(name, args.think)), key=lambda r: (r["task"], r["id"], r["cond"]))
        todo = [x for x in convo_long() + personamem() if (x["task"], x["id"], x["cond"]) not in w.done
                and (not args.tasks or x["task"] in args.tasks.split(","))]
        print(name, "todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        for x in tqdm(todo, desc=name):
            p = chat_prompt(tok, x["user"], think=args.think)
            n = len(tok(p, add_special_tokens=False).input_ids)
            with sdpa_kernel(LONG_SDPA, set_priority=True):
                if args.think:                                        # Qwen3 recommended sampling (10-06)
                    text = generate_think(tok, model, p, args.budget or 1024, (x["task"], x["id"], x["cond"]))
                else:
                    text = generate(tok, model, p, args.budget or (16 if x["task"] == "personamem" else 64))
            # reasoning-trained readers (E13): the answer is the text after the last "Answer:"; others: the first line
            vis = strip_think(text)                                   # E18 thinking mode: the visible reply only
            resp = vis.rsplit("Answer:", 1)[1].strip().split("\n")[0] if "Answer:" in vis else vis.strip().split("\n")[0]
            rec = dict(task=x["task"], id=x["id"], cond=x["cond"], n_tok=n, response=resp, full=text)
            if x["task"] == "personamem":
                pred = pm_pred(text)
                rec.update(qtype=x["qtype"], pred=pred, correct=pred == x["gold"])
            w.write(rec)
            torch.cuda.empty_cache()
        w.close()
        del tok, model
        free_gpu()


@torch.no_grad()
def judge(args):
    conv = {x["id"]: x for x in convo_long() if x["cond"] == "chrono"}
    tok = model = None
    for path in sorted(glob.glob(OUT.format("*"))):
        if path.endswith("_judged.jsonl"):
            continue
        rows = [r for r in load_jsonl(path) if r["task"] == "convo_long"]
        w = JsonlAppender(path.replace(".jsonl", "_judged.jsonl"), key=lambda r: (r["task"], r["id"], r["cond"]))
        todo = [r for r in rows if (r["task"], r["id"], r["cond"]) not in w.done]
        print(os.path.basename(path), "judge todo", len(todo), flush=True)
        if todo and model is None:
            tok, model = load_reader("Qwen3-14B")
        for r in tqdm(todo):
            it = conv[r["id"]]
            text = mem.JUDGE.format(d0=it["dates"][0], d1=it["dates"][1], old=it["ev"][0], new=it["ev"][1], q=it["q"],
                                    a=it["ref"], r=r["response"])
            lp = mem.letter_logprobs(tok, model, text)
            lab = "ABC"[max(range(3), key=lambda i: lp[i])]
            w.write(dict(r, label=lab, correct=lab == "A", stale=lab == "B"))
        w.close()
    if model is not None:
        del tok, model
        free_gpu()


def stats(args):
    from analyze_lora import boot
    for path in sorted(glob.glob(OUT.format("*"))):
        if path.endswith("_judged.jsonl"):
            continue
        name = os.path.basename(path)[len("longconv_"):-len(".jsonl")]
        a = pd.DataFrame([r for r in load_jsonl(path) if r["task"] == "personamem"])
        if len(a):  # re-parse stored outputs with pm_pred (rows written before the 10-04 parser fix)
            gold = {(x["id"], x["cond"]): x["gold"] for x in personamem()}
            a["pred"] = [pm_pred(r.get("full", r.response) if isinstance(r.get("full"), str) else r.response)
                         for _, r in a.iterrows()]
            a["correct"] = [p == gold[(i, c)] for p, i, c in zip(a.pred, a.id, a.cond)]
            print(f"   personamem unparsed {a.pred.isna().sum()} / {len(a)}")
        jp = path.replace(".jsonl", "_judged.jsonl")
        b = pd.DataFrame(load_jsonl(jp)) if os.path.exists(jp) else pd.DataFrame()
        df = pd.concat([a, b], ignore_index=True)
        print("=" * 10, name)
        for task, g in df.groupby("task"):
            piv = g.pivot_table(index="id", columns="cond", values="correct", aggfunc="first").dropna()
            if {"chrono", "rev"} <= set(piv.columns):
                d, lo, hi, _ = boot(piv.rev.astype(float), piv.chrono.astype(float))
                extra = f"| stale rev {g[g.cond == 'rev'].stale.mean() * 100:.1f}" if "stale" in g else ""
                print(f"{task}: n={len(piv)} chrono {piv.chrono.mean() * 100:.1f} rev {piv.rev.mean() * 100:.1f} "
                      f"| rev - chrono {d:.1f} [{lo:.1f}, {hi:.1f}] | median tokens {int(g.n_tok.median())} {extra}")
            if task == "personamem":
                for qt, h in g.groupby("qtype"):
                    print("   ", qt, (h.groupby("cond").correct.mean() * 100).round(1).to_dict(), "n", len(h) // 2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "judge", "stats", "check"])
    ap.add_argument("--models", default="Qwen3-4B")
    ap.add_argument("--budget", type=int, default=None, help="new tokens; default 16 (PersonaMem) / 64 (ConvoMem)")
    ap.add_argument("--tasks", default="", help="comma list of convo_long,personamem (default: both)")
    ap.add_argument("--think", action="store_true", help="Qwen3 thinking mode (E18); results go to <tag>+think")
    args = ap.parse_args()
    if args.stage == "check":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        for xs in (convo_long(), personamem()):
            ls = sorted(len(tok(chat_prompt(tok, x["user"])).input_ids) for x in xs[::7])
            print(xs[0]["task"], "items", len(xs) // 2, "| tokens min/med/max", ls[0], ls[len(ls) // 2], ls[-1],
                  "| gold letters" if xs[0]["task"] == "personamem" else "",
                  dict(pd.Series([x.get("gold") for x in xs]).value_counts()) if xs[0]["task"] == "personamem" else "")
    else:
        {"run": run, "judge": judge, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()
