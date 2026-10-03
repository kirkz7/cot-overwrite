"""Exploration E10 (EXPLORE_PLAN.md): replicate the attention-routing fix of Guo et al. 2026 (2609.38866) in our
order-time setting, then (E10b) replace its oracle parser with the model's own probe.

Their fix: at the answer position, add +beta to the attention scores of the current value's tokens and -beta to the old
values' tokens (final layer(s)); the positions come from a rule-based parser, i.e. the method knows which value is
current. Ours here: the same, with the positions from our ground truth (oracle), applied to whole statements:
  email   our synthetic email threads (value sentences), oldest_first / newest_first
  convo   ConvoMem changing-evidence, personas 25-49 (0-24 are only used for the E7 probes; 50-99 held out), two dated
          conversations in either order (the two evidence messages)
  lme     LongMemEval knowledge-update sessions, chronological / newest-first (the evidence turns)
Conditions: none | route (layers and beta chosen on a calibration set of email threads with other seeds, never on the
test items) | reminder (Guo's baseline: the current statement repeated right before the question; also oracle).
The bias is applied to every generated token's attention over the prompt, in the chosen layers only.
usage: python explore_route.py calibrate | run | judge | stats
"""
import argparse
import glob
import json
import os
import random

import pandas as pd
import torch
from tqdm import tqdm

from app_common import JsonlAppender, chat_prompt, first_number, free_gpu, load_jsonl, load_reader
import run_app_logs as logs
import run_app_memory as mem
from explore_probe import CONVO_GLOB

OUT = "results/explore_route_Qwen3-4B.jsonl"
CAL = "results/explore_route_calibration.json"
GRID = [(nl, b) for nl in (1, 4, 8) for b in (1.0, 2.0, 4.0)]
BIAS = {"mask": None, "layers": set()}


def install(model):
    def make(li):
        def hook(mod, args, kwargs):
            if BIAS["mask"] is None or li not in BIAS["layers"]:
                return None
            hs = kwargs.get("hidden_states", args[0] if args else None)
            if hs is None or hs.shape[1] != 1:          # only single-token (answer) steps
                return None
            kw = dict(kwargs)
            kw["attention_mask"] = BIAS["mask"]
            if len(args) >= 3:                           # attention_mask passed positionally
                args = args[:2] + (BIAS["mask"],) + args[3:]
                kw.pop("attention_mask")
            return args, kw
        return hook
    for li, layer in enumerate(model.model.layers):
        layer.self_attn.register_forward_pre_hook(make(li), with_kwargs=True)


def spans_to_tokens(offsets, spans):
    out = []
    for a, b in spans:
        out += [j for j, (x, y) in enumerate(offsets) if x < b and y > a]
    return out


@torch.no_grad()
def answer(tok, model, prompt, cur, old, route, max_new=48, stops=()):
    """Greedy answer. route = (n_last_layers, beta) or None; cur / old = token positions in the prompt."""
    ids = tok(prompt, add_special_tokens=False).input_ids
    L = model.config.num_hidden_layers
    out = model(torch.tensor([ids[:-1]], device="cuda"), use_cache=True, logits_to_keep=1)
    cache, nxt, gen = out.past_key_values, ids[-1], []
    for step in range(max_new):
        if route:
            n = len(ids) + step
            m = torch.zeros(1, 1, 1, n, device="cuda", dtype=torch.bfloat16)
            m[..., cur] += route[1]
            m[..., old] -= route[1]
            BIAS.update(mask=m, layers=set(range(L - route[0], L)))
        out = model(torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
        BIAS.update(mask=None)
        cache, nxt = out.past_key_values, out.logits[0, -1].argmax().item()
        if nxt in stops:
            break
        gen.append(nxt)
    return tok.decode(gen, skip_special_tokens=True).strip().split("\n")[0]


# ------------------------------------------------------------------ items with statement spans
def email_items(tok, n, seed0, orders=("oldest_first", "newest_first"), reminder=False):
    out = []
    for k in logs.KS:
        for i in range(n):
            it = logs.make_item(k, seed0 + k * 10_000 + i)
            for order in orders:
                user, shown = logs.render(it, "email", order)
                h = it["history"]
                if reminder:
                    sent = f"Quick update: {it['target']} is now {h[-1]}{it['unit']}."
                    q = user.rindex("\n\n")
                    user = user[:q] + f"\n\nReminder - the most recent update says: \"{sent}\"" + user[q:]
                p = chat_prompt(tok, user)
                spans, start = [], 0
                for v in shown:
                    s = f"{it['target']} is now {v}{it['unit']}."
                    a = p.index(s, start); spans.append((a, a + len(s))); start = a + len(s)
                out.append(dict(task="email", id=f"{k}-{i}", cond=order, prompt=p, hist=h,
                                cur=[sp for sp, v in zip(spans, shown) if v == h[-1]],
                                old=[sp for sp, v in zip(spans, shown) if v != h[-1]]))
    return out


def convo_items(tok, personas, per=6, reminder=False):
    files = sorted(glob.glob(CONVO_GLOB))
    out = []
    for pi in personas:
        d = json.load(open(files[pi], encoding="utf-8"))
        for k, ev in enumerate(d["evidence_items"][:per]):
            texts = [m["text"] for m in ev["message_evidences"]]
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
            r = random.Random(f"route{pi}-{k}")
            dates = ["2024-%02d-%02d" % (r.randint(1, 5), r.randint(1, 28)), "2024-%02d-%02d" % (r.randint(7, 11), r.randint(1, 28))]
            for cond, order in (("chrono", [0, 1]), ("rev", [1, 0])):
                blocks = [f"### Conversation (date: {dates[o]})\n" + "\n".join(f"{m['speaker']}: {m['text']}" for m in convs[o])
                          for o in order]
                rem = f"Reminder - the most recent relevant message ({dates[1]}): \"{texts[1]}\"\n" if reminder else ""
                user = ("Here are records of your past conversations with the user.\n\n" + "\n\n".join(blocks) +
                        f"\n\nCurrent date: 2024-12-01\n{rem}Based on the information above, answer the user's question in "
                        f"one short sentence.\nQuestion: {ev['question']}")
                p = chat_prompt(tok, user)
                sp = [(p.index(f"User: {t}"), p.index(f"User: {t}") + len(f"User: {t}")) for t in texts]
                out.append(dict(task="convo", id=f"{pi}-{k}", cond=cond, prompt=p, cur=[sp[1]], old=[sp[0]],
                                q=ev["question"], ref=ev["answer"], ev=texts, dates=dates))
    return out


def lme_items(tok, reminder=False):
    out = []
    for it in mem.load_items():
        if not it["has_ev"]:
            continue
        ev = [[t["content"] for t in s if t.get("has_answer")] for s in it["sessions"]]
        for cond in ("S_chrono_dated", "S_rev_dated"):
            user = mem.build(it, cond)
            if reminder:
                q = user.rindex("Current date:")
                user = user[:q] + f"Reminder - the most recent relevant message ({it['dates'][1]}): \"{it['new_ev']}\"\n" + user[q:]
            p = chat_prompt(tok, user)
            spans = [[(p.index(t), p.index(t) + len(t)) for t in e if t in p] for e in ev]
            out.append(dict(task="lme", id=it["qid"], cond=cond, prompt=p, cur=spans[1], old=spans[0]))
    return out


def stops_of(tok):
    v = tok.get_vocab()
    return {tok.eos_token_id} | {v[t] for t in ("<|im_end|>",) if t in v}


def run_items(tok, model, items, route, label, w):
    stops = stops_of(tok)
    for it in tqdm(items, desc=label):
        if (it["task"], it["id"], it["cond"], label) in w.done:
            continue
        enc = tok(it["prompt"], add_special_tokens=False, return_offsets_mapping=True)
        cur, old = spans_to_tokens(enc.offset_mapping, it["cur"]), spans_to_tokens(enc.offset_mapping, it["old"])
        r = answer(tok, model, it["prompt"], cur, old, route, max_new=24 if it["task"] == "email" else 64, stops=stops)
        rec = dict(task=it["task"], id=it["id"], cond=it["cond"], method=label, response=r)
        if it["task"] == "email":
            pred = first_number(r)
            rec.update(correct=pred == it["hist"][-1], stale=pred in it["hist"][:-1])
        w.write(rec)
        if len(enc.input_ids) > 3000:
            torch.cuda.empty_cache()


def calibrate(args):
    """Choose (layers, beta) on email threads with seeds disjoint from the test threads (newest-first and oldest-first,
    so that a config that breaks correct answers is not chosen). Score = accuracy summed over both orders."""
    tok, model = load_reader("Qwen3-4B")
    install(model)
    items = email_items(tok, 10, 15_000_000)
    stops = stops_of(tok)
    res = {}
    for route in [None] + GRID:
        ok = []
        for it in items:
            enc = tok(it["prompt"], add_special_tokens=False, return_offsets_mapping=True)
            r = answer(tok, model, it["prompt"], spans_to_tokens(enc.offset_mapping, it["cur"]),
                       spans_to_tokens(enc.offset_mapping, it["old"]), route, 24, stops)
            ok.append(first_number(r) == it["hist"][-1])
        res[str(route)] = 100 * sum(ok) / len(ok)
        print(route, round(res[str(route)], 1), flush=True)
    best = max(GRID, key=lambda g: res[str(g)])
    json.dump(dict(scores=res, best=best), open(CAL, "w"))
    print("chosen", best)


def run(args):
    best = tuple(json.load(open(CAL))["best"])
    tok, model = load_reader("Qwen3-4B")
    install(model)
    w = JsonlAppender(OUT, key=lambda r: (r["task"], r["id"], r["cond"], r["method"]))
    plain = email_items(tok, 50, 7_000_000) + convo_items(tok, range(25, 50)) + lme_items(tok)
    rem = email_items(tok, 50, 7_000_000, reminder=True) + convo_items(tok, range(25, 50), reminder=True) + \
        lme_items(tok, reminder=True)
    run_items(tok, model, plain, None, "none", w)
    run_items(tok, model, plain, best, "route", w)
    run_items(tok, model, rem, None, "reminder", w)
    w.close()


JUDGE_C = mem.JUDGE


@torch.no_grad()
def judge(args):
    tok_ = None
    rows = [r for r in load_jsonl(OUT) if r["task"] != "email"]
    w = JsonlAppender(OUT.replace(".jsonl", "_judged.jsonl"), key=lambda r: (r["task"], r["id"], r["cond"], r["method"]))
    todo = [r for r in rows if (r["task"], r["id"], r["cond"], r["method"]) not in w.done]
    print("judge todo", len(todo), flush=True)
    if todo:
        tok, model = load_reader("Qwen3-14B")
        lme = {it["qid"]: it for it in mem.load_items()}
        conv = {}
        for it in convo_items(tok, range(25, 50)):
            conv[it["id"]] = it
        for r in tqdm(todo):
            if r["task"] == "lme":
                it = lme[r["id"]]
                text = JUDGE_C.format(d0=it["dates"][0], d1=it["dates"][1], old=it["old_ev"], new=it["new_ev"],
                                      q=it["question"], a=it["answer"], r=r["response"])
            else:
                it = conv[r["id"]]
                text = JUDGE_C.format(d0=it["dates"][0], d1=it["dates"][1], old=it["ev"][0], new=it["ev"][1],
                                      q=it["q"], a=it["ref"], r=r["response"])
            lp = mem.letter_logprobs(tok, model, text)
            lab = "ABC"[max(range(3), key=lambda i: lp[i])]
            w.write(dict(r, label=lab, correct=lab == "A", stale=lab == "B"))
        del tok, model
        free_gpu()
    w.close()


def stats(args):
    a = pd.DataFrame([r for r in load_jsonl(OUT) if r["task"] == "email"])
    b = pd.DataFrame(load_jsonl(OUT.replace(".jsonl", "_judged.jsonl"))) if os.path.exists(OUT.replace(".jsonl", "_judged.jsonl")) else pd.DataFrame()
    df = pd.concat([a, b], ignore_index=True)
    print("calibration:", json.load(open(CAL)))
    t = (df.groupby(["task", "method", "cond"])[["correct", "stale"]].mean() * 100).round(1)
    print(t.assign(n=df.groupby(["task", "method", "cond"]).size()).to_string())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["calibrate", "run", "judge", "stats", "check"])
    args = ap.parse_args()
    if args.stage == "check":                 # structure only (tokenizer); prints no LongMemEval text
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        for its in (email_items(tok, 5, 7_000_000), convo_items(tok, range(25, 30)), lme_items(tok),
                    email_items(tok, 5, 7_000_000, reminder=True), lme_items(tok, reminder=True)):
            enc = [tok(x["prompt"], add_special_tokens=False, return_offsets_mapping=True) for x in its]
            nc = [len(spans_to_tokens(e.offset_mapping, x["cur"])) for e, x in zip(enc, its)]
            no = [len(spans_to_tokens(e.offset_mapping, x["old"])) for e, x in zip(enc, its)]
            print(its[0]["task"], len(its), "items | current tokens min/med", min(nc), sorted(nc)[len(nc) // 2],
                  "| old tokens min/med", min(no), sorted(no)[len(no) // 2])
    else:
        {"calibrate": calibrate, "run": run, "judge": judge, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()
