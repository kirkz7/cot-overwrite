"""Exp 17: answer extraction / monitoring. A reader is shown a finished solution (not its own
<think>) and asked which final answer it arrives at -- the setting of answer extractors, graders
and CoT monitors. Same competing-conclusion contexts as Exp 13. Also a prompt-level mitigation.

readers:
  extractor  : "What final answer does this solution arrive at?"
  mitigated  : + "the solution may revise itself ... give the answer supported by the reasoning,
                  not simply the last one stated"
usage: python run_exp17.py --n 300
"""
import argparse
import json

import pandas as pd
import torch
from tqdm import tqdm

from early_answer import parse
from probe import greedy, load
from run_exp13 import build_items, keep_deriv

FRACS = [0.3, 1.0]
ASK = "Below is a solution to a math problem. What final answer does the solution arrive at?"
MITIGATE = (" Note: the solution may revise itself or state conflicting conclusions. Give the answer that is"
            " actually supported by the reasoning, not simply the last one stated.")


def prompt(tok, problem, paras, mitigated):
    user = (ASK + (MITIGATE if mitigated else "") + " Reply with the number only.\n\n"
            f"Problem:\n{problem}\n\nSolution:\n" + "\n\n".join(paras))
    head = tok.apply_chat_template([{"role": "user", "content": user}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=False)
    return head + "The final answer is $\\boxed{"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--four_bit", action="store_true")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]
    items = build_items(args.n)
    tok, model = load(args.model, args.four_bit)
    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
    out_path = f"results/exp17_{tag}.jsonl"
    with open(out_path, "w", encoding="utf-8") as f:
        for j, (r, x, pg, px, deriv) in enumerate(tqdm(items)):
            g = r["gold"]
            for fr in FRACS:
                base = keep_deriv(deriv, j, fr)
                for order, tail in (("gold_last", [px, pg]), ("x_last", [pg, px])):
                    for reader, mit in (("extractor", False), ("mitigated", True)):
                        ids = tok(prompt(tok, r["problem"], base + tail, mit), return_tensors="pt",
                                  add_special_tokens=False).input_ids.cuda()
                        pred = parse(tok.decode(greedy(model, ids, 12, stop_ids), skip_special_tokens=True))
                        f.write(json.dumps(dict(uuid=r["uuid"], reader=reader, frac=fr, order=order, gold=g, x=x,
                                                pred=pred, correct=pred == g, picked_x=pred == x)) + "\n")
    df = pd.read_json(out_path, lines=True)
    t = df.groupby(["reader", "frac", "order"]).correct.mean().unstack("order") * 100
    t["position_effect"] = t.gold_last - t.x_last
    t["picked_x_when_x_last"] = df[df.order == "x_last"].groupby(["reader", "frac"]).picked_x.mean() * 100
    print(t.round(1).to_string())


if __name__ == "__main__":
    main()
