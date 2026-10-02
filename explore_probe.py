"""Exploration E7 (EXPLORE_PLAN.md): does the model internally know which value is newer?

Our own synthetic data only (no LongMemEval), two values of the queried quantity (k = 1):
  cot    Exp 18 task, trace shuffled with "Step i:" tags (shuf_step); the newer value is presented first or last at random
  email  run_app_logs.py email threads with dates; oldest_first or newest_first at random
Hidden states (every 2nd layer + the last) are saved at
  decision  the last prompt token (where the answer is produced)
  binding   the last token of each line / sentence that states a value of the queried quantity
Probes (logistic regression, split by item, CPU):
  decision  "is the newer value the one presented first?"   -> also scored on the items the model answers stale
  binding   "is this mention the newer one?"  vs the control "is this mention the last one presented?"
Exploratory; the threshold is written in EXPLORE_PLAN.md before running.
usage: python explore_probe.py run [--models Qwen3-4B,Qwen3-4B@runs/q4-dec-s0/final,...] [--n 600]
       python explore_probe.py stats
"""
import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

from app_common import tag, answer_fast, chat_prompt, first_number, free_gpu, load_reader
import run_app_logs as logs
from probe import Scorer
from tasks import make_example
from tasks_cue import build_cue_prompt, ordered_lines

DIR = "results/probe_e7"


def cot_item(i):
    ex = make_example(1, seed=11_000_000 + i)
    return ex


def email_item(i):
    it = logs.make_item(1, 12_000_000 + i)
    order = "newest_first" if random.Random(i).random() < 0.5 else "oldest_first"
    return it, order


def token_index(offsets, char_end):
    """Index of the token that contains character position char_end - 1."""
    for j, (a, b) in enumerate(offsets):
        if a <= char_end - 1 < b:
            return j
    raise ValueError("char not found in offsets")


@torch.no_grad()
def hidden_at(model, ids, positions, layers):
    out = model(torch.tensor([ids], device="cuda"), output_hidden_states=True, logits_to_keep=1)
    hs = out.hidden_states
    return np.stack([np.stack([hs[l][0, p].float().cpu().numpy() for l in layers]) for p in positions]).astype(np.float16)


def run(args):
    os.makedirs(DIR, exist_ok=True)
    for name in args.models.split(","):
        path = f"{DIR}/{tag(name)}.npz"
        if os.path.exists(path):
            print("exists", path)
            continue
        tok, model = load_reader(name)
        L = model.config.num_hidden_layers
        layers = list(range(0, L + 1, 2)) + ([L] if L % 2 else [])
        scorer = Scorer(tok, model)
        dec_h, dec_meta, bind_h, bind_meta = [], [], [], []
        for i in tqdm(range(args.n), desc=f"{name} cot"):
            ex = cot_item(i)
            prompt, seen = build_cue_prompt(ex, ex.target, "shuf_step", tok)
            hist = ex.history(ex.target)
            s = scorer.score(prompt)
            pred = max(s, key=s.get)
            enc = tok(prompt, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc.input_ids, enc.offset_mapping
            # the two lines of the queried variable, in presented order
            pairs = [(st, l) for st, l in ordered_lines(ex, "shuf_step") if l.var == ex.target]
            pos = []
            for st, l in pairs:
                line = f"Step {st}: {l.bare}"
                c = prompt.rindex(line) + len(line)
                pos.append(token_index(offs, c))
            h = hidden_at(model, ids, [len(ids) - 1] + pos, layers)
            newer_first = seen[0] == hist[-1]
            dec_h.append(h[0]); dec_meta.append(dict(task="cot", i=i, newer_first=newer_first,
                                                     correct=pred == hist[-1], stale=pred == hist[0]))
            for j, (st, l) in enumerate(pairs):
                bind_h.append(h[1 + j]); bind_meta.append(dict(task="cot", i=i, is_newer=l.value == hist[-1],
                                                               is_last_shown=j == len(pairs) - 1))
        for i in tqdm(range(args.n), desc=f"{name} email"):
            it, order = email_item(i)
            user, shown = logs.render(it, "email", order)
            p = chat_prompt(tok, user)
            pred = first_number(answer_fast(tok, model, p))
            h_hist = it["history"]
            enc = tok(p, add_special_tokens=False, return_offsets_mapping=True)
            ids, offs = enc.input_ids, enc.offset_mapping
            pos, start = [], 0
            for v in shown:
                sent = f"{it['target']} is now {v}{it['unit']}."
                c = p.index(sent, start) + len(sent)
                start = c
                pos.append(token_index(offs, c))
            h = hidden_at(model, ids, [len(ids) - 1] + pos, layers)
            dec_h.append(h[0]); dec_meta.append(dict(task="email", i=i, newer_first=shown[0] == h_hist[-1],
                                                     correct=pred == h_hist[-1], stale=pred == h_hist[0]))
            for j, v in enumerate(shown):
                bind_h.append(h[1 + j]); bind_meta.append(dict(task="email", i=i, is_newer=v == h_hist[-1],
                                                               is_last_shown=j == len(shown) - 1))
        np.savez_compressed(path, dec_h=np.stack(dec_h), bind_h=np.stack(bind_h), layers=np.array(layers))
        pd.DataFrame(dec_meta).to_json(path.replace(".npz", "_dec.jsonl"), orient="records", lines=True)
        pd.DataFrame(bind_meta).to_json(path.replace(".npz", "_bind.jsonl"), orient="records", lines=True)
        del tok, model, scorer
        free_gpu()


class Probe:
    """Standardized, L2-regularized logistic regression (LBFGS, CPU); no sklearn dependency."""

    def __init__(self, X, y, l2=1e-2, k=128):
        X = torch.tensor(X, dtype=torch.float32)
        self.mu, self.sd = X.mean(0), X.std(0) + 1e-4
        Z = (X - self.mu) / self.sd
        _, _, V = torch.pca_lowrank(Z, q=min(k, Z.shape[0] - 1), center=False)   # fit on the training items only
        self.V = V
        Z, t = Z @ V, torch.tensor(y, dtype=torch.float32)
        self.w = torch.zeros(Z.shape[1], requires_grad=True)
        self.b = torch.zeros(1, requires_grad=True)
        opt = torch.optim.LBFGS([self.w, self.b], max_iter=200, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(Z @ self.w + self.b, t) + l2 * (self.w ** 2).sum()
            loss.backward()
            return loss
        opt.step(closure)

    def predict(self, X):
        Z = ((torch.tensor(X, dtype=torch.float32) - self.mu) / self.sd) @ self.V
        with torch.no_grad():
            return ((Z @ self.w + self.b) > 0).int().numpy()


def probe_curve(X, y, train, test):
    """Per-layer held-out accuracy; X: [n, layers, d]."""
    accs, models = [], []
    for li in range(X.shape[1]):
        p = Probe(X[train, li].astype(np.float32), y[train])
        accs.append((p.predict(X[test, li].astype(np.float32)) == y[test]).mean() * 100)
        models.append(p)
    return np.array(accs), models


def stats(args):
    import glob
    for path in sorted(glob.glob(f"{DIR}/*.npz")):
        name = os.path.basename(path)[:-4]
        z = np.load(path)
        layers = z["layers"]
        dm = pd.read_json(path.replace(".npz", "_dec.jsonl"), lines=True)
        bm = pd.read_json(path.replace(".npz", "_bind.jsonl"), lines=True)
        print("=" * 20, name)
        for task in ("cot", "email"):
            d = dm[dm.task == task].reset_index(drop=True)
            X = z["dec_h"][dm.index[dm.task == task]]
            y = d.newer_first.values.astype(int)
            rng = np.random.RandomState(0)
            idx = rng.permutation(len(d)); cut = int(0.6 * len(d))
            train, test = idx[:cut], idx[cut:]
            accs, models = probe_curve(X, y, train, test)
            best = int(accs.argmax())
            te = d.iloc[test]
            pred = models[best].predict(X[test, best].astype(np.float32))
            ok = pred == y[test]
            print(f"[{task}] behaviour: correct {d.correct.mean()*100:.1f}%  stale {d.stale.mean()*100:.1f}%  "
                  f"(newer shown first: {d.newer_first.mean()*100:.0f}%)")
            print(f"  decision probe 'newer presented first?': best layer {layers[best]} acc {accs[best]:.1f}% "
                  f"| on items answered stale {ok[te.stale.values].mean()*100:.1f}% (n={te.stale.sum()}) "
                  f"| on items answered correctly {ok[te.correct.values].mean()*100:.1f}% (n={te.correct.sum()})")
            print("  by layer:", dict(zip(layers.tolist(), accs.round(0).astype(int).tolist())))
            b = bm[bm.task == task].reset_index(drop=True)
            Xb = z["bind_h"][bm.index[bm.task == task]]
            items = b.i.unique(); rng.shuffle(items)
            tr_items = set(items[:int(0.6 * len(items))])
            trb = np.where(b.i.isin(tr_items))[0]; teb = np.where(~b.i.isin(tr_items))[0]
            for lab in ("is_newer", "is_last_shown"):
                a, _ = probe_curve(Xb, b[lab].values.astype(int), trb, teb)
                print(f"  binding probe '{lab}': best layer {layers[int(a.argmax())]} acc {a.max():.1f}% "
                      f"| by layer {dict(zip(layers.tolist(), a.round(0).astype(int).tolist()))}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "stats"])
    ap.add_argument("--models", default="Qwen3-4B,Qwen3-4B@runs/q4-dec-s0/final,Qwen3-4B@runs/q4-chr-s0/final")
    ap.add_argument("--n", type=int, default=600)
    args = ap.parse_args()
    {"run": run, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()
