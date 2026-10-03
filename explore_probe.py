"""Exploration E7 (EXPLORE_PLAN.md): does the model internally know which value is newer?

Two values of the queried quantity per item; which one is newer is decided by an explicit marker, and the newer value
is presented first or last at random, so presentation position carries no information about the label.
  cot        Exp 18 task, trace shuffled with "Step i:" tags (shuf_step)                    synthetic (our CoT origin)
  email/git/changelog  run_app_logs.py threads with dates, oldest_first or newest_first     synthetic applications
  convo      ConvoMem changing-evidence (personas 0-49; 50-99 held out), two dated conversations in random order
  mab        MemoryAgentBench FactConsolidation conflict pairs (Wikidata facts, larger serial number = newer), shown
             with 8 other facts of the same context in random order                          real facts
  lme        LongMemEval knowledge-update sessions, chronological vs newest-first           TRANSFER TEST ONLY: no
             probe is ever trained on it; it is only used to score probes trained on the tasks above
Hidden states (every 3rd layer + the last) are saved at the last prompt token (decision) and at the last token of each
statement of the queried quantity (binding).
Probes (scikit-learn logistic regression with cross-validated L2, CPU):
  decision  "is the newer value the one presented first?"   scored also on the items the model answers stale
  binding   "is this statement the newer one?"   vs the control "is this statement the last one presented?"
  transfer  trained on all other tasks (layer chosen by cross-validation inside the training tasks), tested on a
            held-out task and on lme
Exploratory; the threshold is written in EXPLORE_PLAN.md before running. No LongMemEval text is printed.
usage: python explore_probe.py run [--models ...] [--n 400]
       python explore_probe.py stats
"""
import argparse
import glob
import os
import random
import re
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm
import transformers.integrations.sdpa_attention as hf_sdpa
from torch.nn.attention import SDPBackend, sdpa_kernel

# Long prompts (LongMemEval ~9k, padded traces ~19k tokens): repeat K/V instead of SDPA's enable_gqa, so the
# memory-efficient kernel is used; the math fallback needs O(L^2) memory and ran out of memory on 2026-10-02.
hf_sdpa.use_gqa_in_sdpa = lambda *a, **k: False
LONG_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]

from app_common import tag, answer_fast, chat_prompt, first_number, free_gpu, load_jsonl, load_reader
import run_app_logs as logs
import run_app_memory as mem
from probe import Scorer
from tasks import make_example
from tasks_cue import build_cue_prompt, ordered_lines

DIR = "results/probe_e7"
TASKS = ["cot", "email", "git", "changelog", "mab", "convo"]


# ------------------------------------------------------------------ items: (prompt, char ends of the statements, labels)
def cot_items(tok, n):
    out = []
    for i in range(n):
        ex = make_example(1, seed=11_000_000 + i)
        prompt, seen = build_cue_prompt(ex, ex.target, "shuf_step", tok)
        hist = ex.history(ex.target)
        ends = []
        for st, l in [(st, l) for st, l in ordered_lines(ex, "shuf_step") if l.var == ex.target]:
            line = f"Step {st}: {l.bare}"
            ends.append(prompt.rindex(line) + len(line))
        out.append(dict(task="cot", i=i, prompt=prompt, ends=ends, newer_first=seen[0] == hist[-1],
                        is_newer=[v == hist[-1] for v in seen], ex=ex))
    return out


LOG_SENT = {"email": "{k} is now {v}{u}.", "git": "Set {k} to {v}{u}", "changelog": "`{k}` is now {v}{u}."}


def log_items(tok, fmt, n):
    out = []
    for i in range(n):
        it = logs.make_item(1, 12_000_000 + 1000 * TASKS.index(fmt) + i)
        order = "newest_first" if random.Random(f"{fmt}{i}").random() < 0.5 else "oldest_first"
        user, shown = logs.render(it, fmt, order)
        p = chat_prompt(tok, user)
        ends, start = [], 0
        for v in shown:
            s = LOG_SENT[fmt].format(k=it["target"], v=v, u=it["unit"])
            c = p.index(s, start) + len(s)
            ends.append(c); start = c
        h = it["history"]
        out.append(dict(task=fmt, i=i, prompt=p, ends=ends, newer_first=shown[0] == h[-1],
                        is_newer=[v == h[-1] for v in shown], hist=h))
    return out


def mab_items(tok, n):
    from run_ext_eval import MAB_INSTR, mab_items as load_mab
    contexts = {}
    for it in load_mab():
        contexts.setdefault(it["src"], it["facts"])
    pairs = []
    for src, facts in sorted(contexts.items()):
        groups = defaultdict(list)
        for num, t in facts:
            m = re.match(r"(.+?) (is|are|was|were) (.+)\.$", t)
            if m:
                groups[m.group(1) + " " + m.group(2)].append((num, m.group(3), t))
        for key, v in sorted(groups.items()):
            if len(v) == 2 and v[0][1] != v[1][1]:
                pairs.append((src, key, sorted(v), facts))
    rng = random.Random(0)
    rng.shuffle(pairs)
    out = []
    for i, (src, key, pair, facts) in enumerate(pairs[:n]):
        r = random.Random(i)
        others = r.sample([f for f in facts if f[1] not in (pair[0][2], pair[1][2])], 8)
        shown = r.sample([(num, t) for num, _, t in pair] + others, 10)
        user = MAB_INSTR + "\n\nHere is a list of facts:\n" + "\n".join(f"{num}. {t}" for num, t in shown) + \
            f"\n\nComplete the sentence with the up-to-date fact: {key} ..."
        p = chat_prompt(tok, user, prefix=key)
        older, newer = pair[0], pair[1]                      # larger serial number = newer
        order = [x for x in shown if x[1] in (older[2], newer[2])]
        ends = [p.index(f"{num}. {t}") + len(f"{num}. {t}") for num, t in order]
        out.append(dict(task="mab", i=i, prompt=p, ends=ends, newer_first=order[0][1] == newer[2],
                        is_newer=[t == newer[2] for _, t in order], objs=(newer[1], older[1])))
    return out


CONVO_GLOB = r"D:\hf_cache\hub\datasets--Salesforce--ConvoMem\snapshots\*\core_benchmark\evidence_questions\changing_evidence\2_evidence\*.json"


def convo_items(tok, n, personas=range(0, 50)):
    """ConvoMem changing-evidence items (personas 0-49 only: 50-99 are held out for confirmation). The two conversations
    get dates we assign (older < newer, as in the source) and are shown in random order; each is cut to the evidence
    message and up to 5 messages around it."""
    import json
    files = sorted(glob.glob(CONVO_GLOB))
    per = max(1, n // len(personas))
    out = []
    for pi in personas:
        d = json.load(open(files[pi], encoding="utf-8"))
        for k, ev in enumerate(d["evidence_items"][:per]):
            texts = [m["text"] for m in ev["message_evidences"]]            # [older, newer] in source order
            convs = []
            for c, t in zip(ev["conversations"], texts):
                msgs = c["messages"]
                j = next((x for x, m in enumerate(msgs) if m["text"] == t), None)
                if j is None:
                    break
                lo = max(0, j - 3)
                convs.append(msgs[lo:lo + 6])
            if len(convs) != 2:
                continue
            r = random.Random(f"convo{pi}-{k}")
            dates = ["2024-%02d-%02d" % (r.randint(1, 5), r.randint(1, 28)), "2024-%02d-%02d" % (r.randint(7, 11), r.randint(1, 28))]
            order = [1, 0] if r.random() < 0.5 else [0, 1]
            blocks = []
            for o in order:
                body = "\n".join(f"{m['speaker']}: {m['text']}" for m in convs[o])
                blocks.append(f"### Conversation (date: {dates[o]})\n{body}")
            user = ("Here are records of your past conversations with the user.\n\n" + "\n\n".join(blocks) +
                    f"\n\nCurrent date: 2024-12-01\nBased on the information above, answer the user's question in one short "
                    f"sentence.\nQuestion: {ev['question']}")
            p = chat_prompt(tok, user)
            ends = [p.index(f"User: {texts[o]}") + len(f"User: {texts[o]}") for o in order]
            out.append(dict(task="convo", i=f"{pi}-{k}", prompt=p, ends=ends, newer_first=order[0] == 1,
                            is_newer=[o == 1 for o in order]))
    return out


def lme_items(tok):
    out = []
    for it in mem.load_items():
        for cond in ("S_chrono_dated", "S_rev_dated"):
            out.append(dict(task="lme", i=f"{it['qid']}|{cond}", qid=it["qid"], cond=cond,
                            prompt=chat_prompt(tok, mem.build(it, cond)), ends=[], newer_first=cond == "S_rev_dated",
                            is_newer=[]))
    return out


# ------------------------------------------------------------------ model side
def token_index(offsets, char_end):
    for j, (a, b) in enumerate(offsets):
        if a <= char_end - 1 < b:
            return j
    raise ValueError("char not found in offsets")


@torch.no_grad()
def cont_logprob(model, tok, prompt, cont):
    p = tok(prompt, add_special_tokens=False).input_ids
    c = tok(prompt + cont, add_special_tokens=False).input_ids[len(p):]
    ids = torch.tensor([p + c], device="cuda")
    lp = torch.log_softmax(model(ids).logits[0, len(p) - 1:-1].float(), -1)
    return lp[torch.arange(len(c)), torch.tensor(c, device="cuda")].sum().item()


def behaviour(item, model, tok, scorer):
    t = item["task"]
    if t == "cot":
        ex = item["ex"]
        s = scorer.score(item["prompt"])
        pred = max(s, key=s.get)
        h = ex.history(ex.target)
        return pred == h[-1], pred == h[0]
    if t in LOG_SENT:
        pred = first_number(answer_fast(tok, model, item["prompt"]))
        return pred == item["hist"][-1], pred == item["hist"][0]
    if t == "mab":
        new, old = (cont_logprob(model, tok, item["prompt"], " " + o + ".") for o in item["objs"])
        return new > old, old > new
    return None, None                                        # lme: labels come from the judged app_memory files


def task_items(tok, task, n):
    if task == "cot":
        return cot_items(tok, n)
    if task in LOG_SENT:
        return log_items(tok, task, n)
    return {"mab": mab_items, "convo": convo_items}[task](tok, n) if task != "lme" else lme_items(tok)


@torch.no_grad()
def run(args):
    """Resumable: each (model, task) chunk is saved as soon as it is done; finished chunks are skipped."""
    for name in args.models.split(","):
        mdir = f"{DIR}/{tag(name)}"
        os.makedirs(mdir, exist_ok=True)
        todo = [t for t in TASKS + ["lme"] if not os.path.exists(f"{mdir}/{t}.npz")]
        if not todo:
            print("done", mdir)
            continue
        tok, model = load_reader(name)
        L = model.config.num_hidden_layers
        layers = list(range(0, L + 1, 3)) + ([L] if L % 3 else [])
        scorer = Scorer(tok, model)
        for task in todo:
            dec_h, dec_meta, bind_h, bind_meta = [], [], [], []
            for it in tqdm(task_items(tok, task, args.n), desc=f"{name} {task}"):
                enc = tok(it["prompt"], add_special_tokens=False, return_offsets_mapping=True)
                pos = [len(enc.input_ids) - 1] + [token_index(enc.offset_mapping, c) for c in it["ends"]]
                with sdpa_kernel(LONG_SDPA, set_priority=True):
                    out = model(torch.tensor([enc.input_ids], device="cuda"), output_hidden_states=True, logits_to_keep=1)
                    hs = out.hidden_states
                    h = np.stack([np.stack([hs[l][0, p].float().cpu().numpy() for l in layers])
                                  for p in pos]).astype(np.float16)
                    del out, hs
                    correct, stale = behaviour(it, model, tok, scorer)
                dec_h.append(h[0])
                dec_meta.append(dict(task=task, i=str(it["i"]), newer_first=it["newer_first"], correct=correct,
                                     stale=stale, qid=it.get("qid"), cond=it.get("cond")))
                for j, isn in enumerate(it["is_newer"]):
                    bind_h.append(h[1 + j])
                    bind_meta.append(dict(task=task, i=str(it["i"]), is_newer=isn,
                                          is_last_shown=j == len(it["is_newer"]) - 1))
                if len(enc.input_ids) > 3000:
                    torch.cuda.empty_cache()
            pd.DataFrame(dec_meta).to_json(f"{mdir}/{task}_dec.jsonl", orient="records", lines=True)
            pd.DataFrame(bind_meta).to_json(f"{mdir}/{task}_bind.jsonl", orient="records", lines=True)
            np.savez_compressed(f"{mdir}/{task}.tmp.npz", dec_h=np.stack(dec_h),
                                bind_h=np.stack(bind_h) if bind_h else np.zeros((0, len(layers), h.shape[-1]), np.float16),
                                layers=np.array(layers))
            os.replace(f"{mdir}/{task}.tmp.npz", f"{mdir}/{task}.npz")   # the chunk counts as done only once complete
        del tok, model, scorer
        free_gpu()


def load_model_dir(mdir):
    """Concatenate the saved task chunks of one model -> (z-like dict, dec meta, bind meta)."""
    dec, bind, dms, bms, layers = [], [], [], [], None
    for t in TASKS + ["lme"]:
        if not os.path.exists(f"{mdir}/{t}.npz"):
            continue
        z = np.load(f"{mdir}/{t}.npz")
        layers = z["layers"]
        dec.append(z["dec_h"]); bind.append(z["bind_h"])
        dms.append(pd.read_json(f"{mdir}/{t}_dec.jsonl", lines=True, dtype={"i": str}))
        if os.path.getsize(f"{mdir}/{t}_bind.jsonl") > 2:
            bms.append(pd.read_json(f"{mdir}/{t}_bind.jsonl", lines=True, dtype={"i": str}))
    return (dict(dec_h=np.concatenate(dec), bind_h=np.concatenate(bind), layers=layers),
            pd.concat(dms, ignore_index=True), pd.concat(bms, ignore_index=True))


# ------------------------------------------------------------------ probes (CPU)
def fit(X, y):
    from sklearn.linear_model import LogisticRegressionCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    return make_pipeline(StandardScaler(), LogisticRegressionCV(Cs=[1e-3, 1e-2, 1e-1, 1.0], cv=5, max_iter=3000)).fit(X, y)


def layer_curve(X, y, train, test):
    accs, models = [], []
    for li in range(X.shape[1]):
        m = fit(X[train, li].astype(np.float32), y[train])
        accs.append((m.predict(X[test, li].astype(np.float32)) == y[test]).mean() * 100)
        models.append(m)
    return np.array(accs), models


def pick_layer(X, y, n_layers):
    """Choose the layer by 5-fold accuracy inside the training data only."""
    from sklearn.model_selection import cross_val_score
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    sc = [cross_val_score(make_pipeline(StandardScaler(), LogisticRegression(C=1e-2, max_iter=3000)),
                          X[:, li].astype(np.float32), y, cv=5).mean() for li in range(n_layers)]
    return int(np.argmax(sc))


def lme_labels(name):
    """Judged LongMemEval answers of this reader (correct / stale per qid and condition)."""
    p = f"results/app_memory_{name}_judged.jsonl"
    if not os.path.exists(p):
        return {}
    return {(r["qid"], r["cond"]): (r["label"] == "A", r["label"] == "B") for r in load_jsonl(p) if not r["skipped"]}


def stats(args):
    rows = []
    for mdir in sorted(d for d in glob.glob(f"{DIR}/*") if os.path.isdir(d)):
        name = os.path.basename(mdir)
        z, dm, bm = load_model_dir(mdir)
        layers = z["layers"]
        lab = lme_labels(name)
        for k, (q, c) in enumerate(zip(dm.qid, dm.cond)):
            if dm.task[k] == "lme" and (q, c) in lab:
                dm.loc[k, "correct"], dm.loc[k, "stale"] = lab[(q, c)]
        X, y = z["dec_h"], dm.newer_first.values.astype(int)
        print("=" * 20, name)
        for task in TASKS:
            idx = np.where(dm.task == task)[0]
            rng = np.random.RandomState(0)
            perm = rng.permutation(idx); cut = int(0.6 * len(perm))
            tr, te = perm[:cut], perm[cut:]
            accs, models = layer_curve(X, y, tr, te)
            best = pick_layer(X[tr], y[tr], X.shape[1])          # chosen inside the training split, not on the test items
            pred = models[best].predict(X[te, best].astype(np.float32))
            ok = pred == y[te]
            st = dm.stale.values[te].astype(bool); co = dm.correct.values[te].astype(bool)
            Xb = z["bind_h"]; bidx = np.where(bm.task == task)[0]
            items = np.array(sorted(set(bm.i[bidx]))); rng.shuffle(items)
            trs = set(items[:int(0.6 * len(items))])
            btr = bidx[np.isin(bm.i.values[bidx], list(trs))]
            bte = bidx[~np.isin(bm.i.values[bidx], list(trs))]
            bn, _ = layer_curve(Xb, bm.is_newer.values.astype(int), btr, bte)
            bl, _ = layer_curve(Xb, bm.is_last_shown.values.astype(int), btr, bte)
            rows.append(dict(model=name, task=task, n=len(idx), behav_correct=dm.correct.values[idx].astype(float).mean() * 100,
                             behav_stale=dm.stale.values[idx].astype(float).mean() * 100,
                             dec_layer=int(layers[best]), dec_acc=accs[best], dec_acc_on_stale=ok[st].mean() * 100 if st.any() else np.nan,
                             n_stale_test=int(st.sum()), dec_acc_on_correct=ok[co].mean() * 100 if co.any() else np.nan,
                             bind_newer=bn.max(), bind_newer_layer=int(layers[int(bn.argmax())]), bind_lastshown=bl.max()))
            print(f"  {task}: decision by layer", dict(zip(layers.tolist(), accs.round(0).astype(int).tolist())))
        # transfer: train on the other synthetic / real tasks, test on a held-out task, and on lme (never trained on)
        for held in TASKS + ["lme"]:
            tr = np.where((dm.task != held) & (dm.task != "lme"))[0]
            te = np.where(dm.task == held)[0]
            li = pick_layer(X[tr], y[tr], X.shape[1])
            m = fit(X[tr, li].astype(np.float32), y[tr])
            ok = m.predict(X[te, li].astype(np.float32)) == y[te]
            st = dm.stale.values[te].astype(float) == 1
            rows.append(dict(model=name, task=f"transfer->{held}", n=len(te), dec_layer=int(layers[li]),
                             dec_acc=ok.mean() * 100, dec_acc_on_stale=ok[st].mean() * 100 if st.any() else np.nan,
                             n_stale_test=int(st.sum())))
    df = pd.DataFrame(rows)
    pd.set_option("display.width", 250)
    print(df.round(1).to_string(index=False))
    df.round(2).to_csv(f"{DIR}/summary.csv", index=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "stats", "check"])
    ap.add_argument("--models", default="Qwen3-4B,Qwen3-4B@runs/q4-dec-s0/final,Qwen3-4B@runs/q4-chr-s0/final")
    ap.add_argument("--n", type=int, default=400)
    args = ap.parse_args()
    if args.stage == "check":                 # structure only (tokenizer), prints no LongMemEval text
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        items = cot_items(tok, 50) + [x for f in LOG_SENT for x in log_items(tok, f, 50)] + mab_items(tok, 50) +             convo_items(tok, 400) + lme_items(tok)
        for task in TASKS + ["lme"]:
            its = [x for x in items if x["task"] == task]
            for x in its:
                enc = tok(x["prompt"], add_special_tokens=False, return_offsets_mapping=True)
                [token_index(enc.offset_mapping, c) for c in x["ends"]]
            print(task, "items", len(its), "| newer shown first", round(np.mean([x["newer_first"] for x in its]), 2),
                  "| statements per item", sorted({len(x["ends"]) for x in its}))
        print("mab pairs available:", len(mab_items(tok, 10_000)))
    else:
        {"run": run, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()
