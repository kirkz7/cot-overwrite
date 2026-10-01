"""Exp 22: does order matter in proportion to state updates? All 27 BBH tasks, the model's own CoT.

For each task: generate a CoT (Qwen3-4B, thinking off, greedy, batched), keep correct examples,
strip the answer sentence, then re-read the CoT ordered vs line-shuffled (3 seeds) and decode the
answer. Shuffle damage per task is compared with
  (a) an a-priori label: does solving the task require maintaining a state that gets UPDATED
      (running count, position, stack, object locations, partial results)?
  (b) an automatic overwrite proxy from the CoT: share of lines whose "key" (text before ':' / '='
      / ' is ' / ' has ') repeats an earlier line's key with different content.
Prediction: damage concentrates in state-update tasks (reconciling 2605.26795 / 2605.07307 with
2605.22870, which saw collapse on tracking tasks).

usage: python run_exp22.py --model Qwen/Qwen3-4B
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

TASKS = {  # a-priori: "state" = CoT maintains a state that is updated; "partial"; "none"
    "tracking_shuffled_objects_three_objects": "state", "tracking_shuffled_objects_five_objects": "state",
    "tracking_shuffled_objects_seven_objects": "state", "object_counting": "state", "navigate": "state",
    "dyck_languages": "state", "multistep_arithmetic_two": "state", "boolean_expressions": "state",
    "date_understanding": "partial", "penguins_in_a_table": "partial", "reasoning_about_colored_objects": "partial",
    "word_sorting": "partial", "temporal_sequences": "none", "web_of_lies": "none",
    "logical_deduction_three_objects": "none", "logical_deduction_five_objects": "none",
    "logical_deduction_seven_objects": "none", "geometric_shapes": "none", "causal_judgement": "none",
    "disambiguation_qa": "none", "formal_fallacies": "none", "hyperbaton": "none", "movie_recommendation": "none",
    "ruin_names": "none", "salient_translation_error_detection": "none", "snarks": "none",
    "sports_understanding": "none",
}
# answer type per task (v1 used a literal "<answer>" placeholder that the model copied as "<False>")
KIND = {"boolean_expressions": "bool", "causal_judgement": "yesno", "navigate": "yesno", "web_of_lies": "yesno",
        "sports_understanding": "yesno_lower", "formal_fallacies": "valid", "object_counting": "number",
        "multistep_arithmetic_two": "number", "dyck_languages": "dyck", "word_sorting": "words"}  # others: "mc"
FINISH = {
    "mc": "Finish with a final line of the form: So the answer is (X), where X is the letter of the correct option.",
    "bool": "Finish with a final line that is either: So the answer is True. or: So the answer is False.",
    "yesno": "Finish with a final line that is either: So the answer is Yes. or: So the answer is No.",
    "yesno_lower": "Finish with a final line that is either: So the answer is yes. or: So the answer is no.",
    "valid": "Finish with a final line that is either: So the answer is valid. or: So the answer is invalid.",
    "number": "Finish with a final line of the form: So the answer is N. where N is a single number.",
    "dyck": "Finish with a final line: So the answer is followed by only the closing brackets needed, separated by spaces.",
    "words": "Finish with a final line: So the answer is followed by the sorted words separated by single spaces.",
}
CHOICES = {"bool": [" True", " False"], "yesno": [" Yes", " No"], "yesno_lower": [" yes", " no"],
           "valid": [" valid", " invalid"]}
ANS_LINE = re.compile(r"(?i)\banswer\b|^\W*(so|therefore|thus|hence)\b|^\W*at the end\b")
KEY = re.compile(r"^\W*(?:step \d+\W*)?(.{1,40}?)(?:\s*[:=]|\s+(?:is|has|are|have)\b)", re.I)
GENERIC_KEYS = {"i", "we", "it", "this", "that", "there", "so", "then", "now", "first", "next", "finally", "which",
                "the answer", "answer", "result", "step"}


def kind(task):
    return KIND.get(task, "mc")


def instr(task):
    return "\n\nLet's think step by step. Write one short line per step. " + FINISH[kind(task)]


def chat(tok, user):
    return tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)


def norm(s):
    s = s.strip().lower().replace("**", "")
    s = re.sub(r"[\s,]+", " ", s)
    return s.strip(" .;:'\"`$")


def is_correct(answer, target):
    if not isinstance(answer, str):   # None, or NaN after a pandas round-trip
        return False
    a0 = answer.strip()
    if a0.startswith("<") and a0.rstrip(".").endswith(">") and "<" not in target:
        answer = a0.rstrip(".")[1:-1]
    if re.fullmatch(r"\([A-R]\)", target.strip()):
        m = re.search(r"\(?([A-R])\)", answer) or re.match(r"\s*([A-R])\b", answer)
        return bool(m) and m.group(1) == target.strip()[1]
    a, t = norm(answer), norm(target)
    if t in ("true", "false", "yes", "no", "valid", "invalid"):
        return a.split(" ")[0] == t
    return a.replace(" ", "") == t.replace(" ", "") or a == t


def extract(gen):
    m = re.findall(r"answer is[:\s]*(.+?)\s*(?:\.\s*$|$)", gen, re.I | re.M)
    return m[-1] if m else None


def overwrite_rate(lines):
    seen, over = {}, 0
    for l in lines:
        m = KEY.match(l)
        if not m:
            continue
        k, rest = m.group(1).strip().lower(), l[m.end():].strip().lower()
        if k in GENERIC_KEYS or len(k) < 2:
            continue
        if k in seen and seen[k] != rest:
            over += 1
        seen[k] = rest
    return over / max(len(lines), 1)


def letters_in(question):
    opts = question.split("Options:")[-1] if "Options:" in question else ""
    return re.findall(r"\(([A-R])\)", opts) or list("ABCDE")


@torch.no_grad()
def score_choices(tok, model, prefix, cands):
    """log p of the first token of each candidate continuation (candidates differ at token 1)."""
    p_ids = tok(prefix, add_special_tokens=False).input_ids
    first = {}
    for c in cands:
        full = tok(prefix + c, add_special_tokens=False).input_ids
        assert full[:len(p_ids)] == p_ids, (prefix[-30:], c)
        first[c] = full[len(p_ids)]
    lp = torch.log_softmax(model(torch.tensor([p_ids], device="cuda"), logits_to_keep=1).logits[0, -1].float(), -1)
    return {c: lp[t].item() for c, t in first.items()}


@torch.no_grad()
def batch_generate(tok, model, prompts, bs, max_new):
    tok.padding_side = "left"
    outs = []
    for i in tqdm(range(0, len(prompts), bs)):
        enc = tok(prompts[i:i + bs], return_tensors="pt", padding=True, add_special_tokens=False).to("cuda")
        gen = model.generate(**enc, max_new_tokens=max_new, do_sample=False, temperature=None, top_p=None,
                             top_k=None, pad_token_id=tok.pad_token_id)
        outs += tok.batch_decode(gen[:, enc.input_ids.shape[1]:], skip_special_tokens=True)
    return outs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--seeds", type=int, default=3)
    ap.add_argument("--bs", type=int, default=16)
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    tok, model = load(args.model)
    gen_path = f"results/bbh27_gen_v2_{tag}.jsonl"
    if not os.path.exists(gen_path):
        rows = [dict(task=t, input=ex["input"], target=ex["target"]) for t in TASKS
                for ex in load_dataset("lukaemon/bbh", t, split="test")]
        gens = batch_generate(tok, model, [chat(tok, r["input"] + instr(r["task"])) for r in rows], args.bs, 512)
        with open(gen_path, "w", encoding="utf-8") as f:
            for r, g in zip(rows, gens):
                r.update(cot=g, gen_answer=extract(g))
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
    rows = [json.loads(l) for l in open(gen_path, encoding="utf-8")]
    jobs = []
    for i, r in enumerate(rows):
        if not is_correct(r["gen_answer"], r["target"]):
            continue
        lines = [l.strip() for l in r["cot"].split("\n") if l.strip() and not ANS_LINE.search(l)]
        if len(lines) < 2:
            continue
        ow = overwrite_rate(lines)
        jobs.append((i, r, "full", 0, lines, ow))
        for s in range(1, args.seeds + 1):
            sh = lines[:]
            random.Random(1000 * i + s).shuffle(sh)
            jobs.append((i, r, "shuf", s, sh, ow))
    # readout: closed-set tasks by log-prob over the options; open tasks (number/dyck/words) by greedy decoding
    preds = {}
    open_jobs = [j for j, job in enumerate(jobs) if kind(job[1]["task"]) in ("number", "dyck", "words")]
    for j, (i, r, cond, seed, ls, ow) in enumerate(tqdm(jobs, desc="closed-set readout")):
        k = kind(r["task"])
        body = chat(tok, r["input"]) + "\n".join(ls) + "\n"
        if k == "mc":
            sc = score_choices(tok, model, body + "So the answer is (", letters_in(r["input"]))
            preds[j] = "(" + max(sc, key=sc.get) + ")"
        elif k in CHOICES:
            sc = score_choices(tok, model, body + "So the answer is", CHOICES[k])
            preds[j] = max(sc, key=sc.get).strip()
    outs = batch_generate(tok, model, [chat(tok, jobs[j][1]["input"]) + "\n".join(jobs[j][4]) + "\nSo the answer is "
                                       for j in open_jobs], args.bs, 48)
    for j, o in zip(open_jobs, outs):
        preds[j] = o.split("\n")[0].strip().rstrip(".")
    out_path = f"results/exp22_v2_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for j, (i, r, cond, seed, ls, ow) in enumerate(jobs):
            f.write(json.dumps(dict(task=r["task"], label=TASKS[r["task"]], kind=kind(r["task"]), idx=i, cond=cond,
                                    seed=seed, n_lines=len(ls), overwrite=ow, target=r["target"], pred=preds[j],
                                    correct=is_correct(preds[j], r["target"]))) + "\n")
    df = pd.read_json(out_path, lines=True)
    gen_acc = pd.DataFrame(rows).assign(ok=lambda d: [is_correct(a, t) for a, t in zip(d.gen_answer, d.target)])
    t = df.groupby(["task", "cond"]).correct.mean().unstack("cond") * 100
    t["drop"] = t["full"] - t["shuf"]
    t["label"] = [TASKS[x] for x in t.index]
    t["overwrite"] = df[df.cond == "full"].groupby("task").overwrite.mean()
    t["n"] = df[df.cond == "full"].groupby("task").size()
    t["gen_acc"] = gen_acc.groupby("task").ok.mean() * 100
    print(t.sort_values("drop", ascending=False).round(2).to_string())
    print("\nmean drop by a-priori label:", t.groupby("label")["drop"].mean().round(1).to_dict())
    print("spearman(drop, overwrite proxy):", round(t[["drop", "overwrite"]].corr(method="spearman").iloc[0, 1], 3))


if __name__ == "__main__":
    main()
