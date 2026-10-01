"""Exp 7: where does "last-presented value wins" live?

1. Attention: at the answer position, for every (layer, head), the attention mass on the value
   tokens of each line of the queried variable. A head's "recency score" = mean share of that
   mass on the line presented last, over shuffled k=4 examples.
2. Ablation: zero the outputs of the top-N recency heads (pre-hook on o_proj) and re-score
   shuffled / ordered examples; compare with N random heads.

usage: python mech_exp7.py --model Qwen/Qwen3-4B --n 200
"""
import argparse
import json
import random

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from probe import Scorer
from tasks import build_prompt, make_example, presented_lines


def value_spans(tok, prompt, ex, cond, qv):
    """Token indices of the value digits for each presented line of var qv, in presented order."""
    body_start = prompt.index("Let me trace the program step by step.")
    enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
    offs = enc.offset_mapping
    spans, pos = [], body_start
    for l in presented_lines(ex, cond):
        s = prompt.index(l.bare, pos)
        pos = s + len(l.bare)
        if l.var != qv:
            continue
        c0, c1 = s + len(l.bare) - len(str(l.value)), s + len(l.bare)
        spans.append((l.value, [i for i, (a, b) in enumerate(offs) if a < c1 and b > c0]))
    return enc.input_ids, spans


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen3-4B")
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--topn", default="4,8,16,32")
    args = ap.parse_args()
    tag = args.model.split("/")[-1]

    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda",
                                                 attn_implementation="eager").eval()
    cfg = model.config
    L, H = cfg.num_hidden_layers, cfg.num_attention_heads
    hd = getattr(cfg, "head_dim", cfg.hidden_size // H)
    exs = [make_example(args.k, seed=args.k * 100_000 + i) for i in range(args.n)]

    # ---- 1. attention to value tokens ----
    rec = np.zeros((L, H)); gold_share = np.zeros((L, H)); mass = np.zeros((L, H))
    with torch.no_grad():
        for ex in exs:
            p = build_prompt(ex, ex.target, "shuf", "bare", tok)
            ids, spans = value_spans(tok, p, ex, "shuf", ex.target)
            att = model(torch.tensor([ids], device="cuda"), output_attentions=True).attentions
            gold_i = [v for v, _ in spans].index(ex.history(ex.target)[-1])
            for li, a in enumerate(att):
                last = a[0, :, -1].float()                         # [H, seq]
                per_line = torch.stack([last[:, s].sum(-1) for _, s in spans], -1)  # [H, k+1]
                tot = per_line.sum(-1)
                share = per_line / tot.clamp_min(1e-9)[:, None]
                rec[li] += share[:, -1].cpu().numpy()
                gold_share[li] += share[:, gold_i].cpu().numpy()
                mass[li] += tot.cpu().numpy()
    rec /= args.n; gold_share /= args.n; mass /= args.n
    # rank heads that both look at the values (mass) and prefer the last-presented one
    score = mass * (rec - 1 / (args.k + 1))
    order = np.dstack(np.unravel_index(np.argsort(-score, axis=None), score.shape))[0]
    top = [(int(l), int(h), round(float(mass[l, h]), 3), round(float(rec[l, h]), 3), round(float(gold_share[l, h]), 3))
           for l, h in order[:20]]
    print("top recency heads (layer, head, value-mass, share-on-last, share-on-gold):")
    for t in top:
        print("  ", t)
    np.savez(f"results/exp7_{tag}_attn.npz", rec=rec, gold_share=gold_share, mass=mass, score=score)

    # ---- 2. ablation ----
    del model; torch.cuda.empty_cache()
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map="cuda").eval()
    scorer = Scorer(tok, model)
    ablate = {}  # layer -> list of heads

    def make_hook(li):
        def hook(mod, inp):
            hs = ablate.get(li)
            if not hs:
                return None
            x = inp[0].clone()
            for h in hs:
                x[..., h * hd:(h + 1) * hd] = 0
            return (x,)
        return hook
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.o_proj.register_forward_pre_hook(make_hook(li))

    def evaluate(cond):
        acc = last = 0
        for ex in exs:
            s = scorer.score(build_prompt(ex, ex.target, cond, "bare", tok))
            pred = max(s, key=s.get)
            seen = [l.value for l in presented_lines(ex, cond) if l.var == ex.target]
            acc += pred == ex.history(ex.target)[-1]; last += pred == seen[-1]
        return round(100 * acc / len(exs), 1), round(100 * last / len(exs), 1)

    results = []
    rng = random.Random(0)
    all_heads = [(l, h) for l in range(L) for h in range(H)]
    for n in [0] + [int(x) for x in args.topn.split(",")]:
        for kind in (["top"] if n == 0 else ["top", "random"]):
            heads = [tuple(map(int, x)) for x in order[:n]] if kind == "top" else rng.sample(all_heads, n)
            ablate.clear()
            for l, h in heads:
                ablate.setdefault(l, []).append(h)
            r = dict(n=n, kind=kind, shuf=evaluate("shuf"), full=evaluate("full"))
            print(r, flush=True)
            results.append(r)
    json.dump(dict(top_heads=top, ablation=results), open(f"results/exp7_{tag}.json", "w"), indent=1)


if __name__ == "__main__":
    main()
