"""Exp 20: BBH with the model's own CoT -- does line-shuffling hurt exactly where states are
overwritten? (the tasks on which 2605.22870 reported collapse)

1. Generate a CoT with Qwen3-4B (thinking off, greedy, batched) for
     tracking_shuffled_objects_{three,five,seven}_objects  (each swap overwrites two states)
     logical_deduction_five_objects                         (constraints accumulate; control)
   and keep examples whose generated answer is correct.
2. Probe-time: question + CoT (answer sentence stripped), ordered or line-shuffled (3 seeds),
   read the option letter by log-prob after "So the answer is (".
3. For tracking, k = number of swaps that involve the queried person (state overwrites).

usage: python run_exp20.py --model Qwen/Qwen3-4B
"""
import argparse
import json
import os
import random
import re

import pandas as pd
import torch
from datasets import load_dataset
from tqdm import tqdm

from probe import load

TASKS = ["tracking_shuffled_objects_three_objects", "tracking_shuffled_objects_five_objects",
         "tracking_shuffled_objects_seven_objects", "logical_deduction_five_objects"]
INSTR = ("\n\nLet's think step by step. Write one short line per step, stating the current situation "
         "after that step. Finish with 'So the answer is (X).'")
STEM = "So the answer is ("
# answer / summary sentences only; NOT "Finally," -- BBH words the last swap that way and the CoT echoes it
ANS_LINE = re.compile(r"(?i)\banswer\b|^\W*(so|therefore|thus|hence)\b|^\W*at the end\b")
SWAP = re.compile(r"([A-Z][a-z]+) and ([A-Z][a-z]+) (?:swap|switch|trade)")
QUERY = re.compile(r"At the end of the [^,]+, ([A-Z][a-z]+) (?:has|is)")


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


@torch.no_grad()
def generate_all(tok, model, prompts, bs=16, max_new=512):
    tok.padding_side = "left"
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    outs = []
    for i in tqdm(range(0, len(prompts), bs), desc="generate"):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def letters_in(question):
    return re.findall(r"\(([A-G])\)", question.split("Options:")[-1]) if "Options:" in question else list("ABCDE")


@torch.no_grad()
def score_letters(tok, model, prefix, letters):
    p_ids = tok(prefix, add_special_tokens=False).input_ids
    nxt = {}
    for L in letters:
        full = tok(prefix + L, add_special_tokens=False).input_ids
        assert full[:len(p_ids)] == p_ids, "prefix tokenization changed"
        nxt[L] = full[len(p_ids)]
    logits = model(torch.tensor([p_ids], device="cuda"), logits_to_keep=1).logits[0, -1].float()
    lp = torch.log_softmax(logits, -1)
    return {L: lp[t].item() for L, t in nxt.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--seeds", type=int, default=3)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    tok, model = load(args.model)
    gen_path = f"results/bbh_gen_{tag}.jsonl"
    if not os.path.exists(gen_path):
        rows = []
        for t in TASKS:
            for ex in load_dataset("lukaemon/bbh", t, split="test"):
                rows.append(dict(task=t, input=ex["input"], target=ex["target"].strip("()")))
        gens = generate_all(tok, model, [chat(tok, r["input"] + INSTR) for r in rows])
        with open(gen_path, "w", encoding="utf-8") as f:
            for r, g in zip(rows, gens):
                m = re.findall(r"answer is \(?([A-G])\)?", g)
                r.update(cot=g, gen_answer=m[-1] if m else None)
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [json.loads(l) for l in open(gen_path, encoding="utf-8")]
    out_path = f"results/exp20_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for i, r in enumerate(tqdm(rows, desc="probe")):
            if r["gen_answer"] != r["target"]:
                continue
            lines = [l.strip() for l in r["cot"].split("\n") if l.strip()]
            lines = [l for l in lines if not ANS_LINE.search(l)]
            if len(lines) < 2:
                continue
            letters = letters_in(r["input"])
            head = chat(tok, r["input"])
            k = None
            if r["task"].startswith("tracking"):
                q = QUERY.search(r["input"])
                if q:
                    k = sum(q.group(1) in pair for pair in SWAP.findall(r["input"]))
            conds = [("full", 0, lines)]
            for s in range(1, args.seeds + 1):
                sh = lines[:]
                random.Random(1000 * i + s).shuffle(sh)
                conds.append(("shuf", s, sh))
            for cond, seed, ls in conds:
                sc = score_letters(tok, model, head + "\n".join(ls) + "\n" + STEM, letters)
                pred = max(sc, key=sc.get)
                f.write(json.dumps(dict(task=r["task"], idx=i, cond=cond, seed=seed, k=k, n_lines=len(lines),
                                        target=r["target"], pred=pred, correct=pred == r["target"])) + "\n")
    df = pd.read_json(out_path, lines=True)
    print("generated-answer accuracy:", pd.DataFrame(rows).assign(ok=lambda d: d.gen_answer == d.target)
          .groupby("task").ok.mean().round(3).to_dict())
    print((df.groupby(["task", "cond"]).correct.mean().unstack("cond") * 100).round(1).to_string())
    tr = df[df.task.str.startswith("tracking")]
    print("\ntracking, by swaps involving the queried person (k):")
    print((tr.groupby(["k", "cond"]).correct.mean().unstack("cond") * 100).round(1)
          .assign(n=tr[tr.cond == "full"].groupby("k").size()).to_string())


if __name__ == "__main__":
    main()
