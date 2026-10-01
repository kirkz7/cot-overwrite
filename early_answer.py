"""Exp 6 step A: early answering (Lanham et al. 2023) on real R1 traces.

For each trace, the probe answers after reading the first 10%, 20%, ..., 100% of the
paragraphs. One full prefill per trace; truncation points reuse the KV cache via crop(),
visited from the latest to the earliest point.

Output: results/early_<model>.jsonl with the probe's answer at each truncation point.
usage: python early_answer.py --model Qwen/Qwen3-4B --n 1500
"""
import argparse
import json
import os
import random
import re

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from tqdm import tqdm

from probe import PREFILL, load

INT = re.compile(r"(?<![\d.])-?\d+(?!\d|\.\d)")
FRACS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
SUFFIX = "\n</think>\n\nThe final answer is $\\boxed{"


def segments(tok, problem, paras):
    """Token ids for [head+<think>, para_1, '\\n\\n'+para_2, ...]; concatenated, never re-tokenized jointly."""
    head = tok.apply_chat_template([{"role": "user", "content": problem}], tokenize=False,
                                   add_generation_prompt=True, enable_thinking=True).split("<think>")[0]
    enc = lambda s: tok(s, add_special_tokens=False).input_ids
    return [enc(head + "<think>\n")] + [enc(("\n\n" if i else "") + p) for i, p in enumerate(paras)]


def parse(txt):
    m = INT.search(txt.split("}")[0].replace(",", "").replace(" ", ""))
    return int(m.group()) if m else None


@torch.no_grad()
def answer_from_cache(model, cache, suffix_ids, stop_ids, max_new=12):
    """Append suffix to the cache and greedy-decode. The caller crops the cache afterwards."""
    with sdpa_kernel(SDPBackend.MATH):
        out = model(torch.tensor([suffix_ids], device="cuda"), past_key_values=cache, use_cache=True)
        gen, nxt = [], out.logits[0, -1].argmax().item()
        for _ in range(max_new):
            gen.append(nxt)
            if nxt in stop_ids:
                break
            out = model(torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
            nxt = out.logits[0, -1].argmax().item()
    return gen


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=1500)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open("data_exp4.jsonl", encoding="utf-8")]
    random.Random(1).shuffle(rows)
    rows = rows[:args.n]
    tok, model = load(args.model)
    stop_ids = {i for t, i in tok.get_vocab().items() if "}" in tok.convert_tokens_to_string([t])} | {tok.eos_token_id}
    suffix_ids = tok(SUFFIX, add_special_tokens=False).input_ids
    tag = args.model.split("/")[-1]
    os.makedirs("results", exist_ok=True)

    with open(f"results/early_{tag}.jsonl", "w", encoding="utf-8") as f:
        for r in tqdm(rows):
            segs = segments(tok, r["problem"], r["paras"])
            ids = [t for s in segs for t in s]
            ends = [sum(map(len, segs[:j + 1])) for j in range(len(segs))]  # token end of head, para1, ...
            n_p = len(r["paras"])
            points = sorted({max(1, round(fr * n_p)) for fr in FRACS})
            with torch.no_grad(), sdpa_kernel(PREFILL, set_priority=True):
                cache = model(torch.tensor([ids], device="cuda"), use_cache=True, logits_to_keep=1).past_key_values
            answers = {}
            for j in sorted(points, reverse=True):
                cache.crop(ends[j])
                gen = answer_from_cache(model, cache, suffix_ids, stop_ids)
                answers[j] = parse(tok.decode(gen, skip_special_tokens=True))
            del cache
            seen_text = lambda j: "\n\n".join(r["paras"][:j])
            f.write(json.dumps(dict(
                uuid=r["uuid"], group=r["group"], gold=r["gold"], n_paras=n_p,
                points=points, answers=[answers[j] for j in points],
                # was the early answer literally written in the trace before that point?
                in_text=[answers[j] is not None and str(answers[j]) in set(INT.findall(seen_text(j))) for j in points],
            )) + "\n")


if __name__ == "__main__":
    main()
