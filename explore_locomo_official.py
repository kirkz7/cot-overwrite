"""LoCoMo end to end, the official way (user 10-08 09:55; CLOUD_NOTEBOOK "LoCoMo 官方端到端", written before running).
Uses the official evaluation code of snap-research/locomo (commit 3eb6f2c5, copied to /data/locomo_official):
  context   task_eval/hf_llm_utils.get_input_context + CONV_START_PROMPT (the open-model path). NOTE: this function
            puts the sessions NEWEST FIRST (turns inside a session stay in order); the GPT path does the same.
  question  QA_PROMPT; temporal questions (category 2) get " Use DATE of CONVERSATION to answer with an approximate date."
  chat      one user message = context + "\n\n" + question prompt (the Mistral / Gemma variant: no system prompt)
  decoding  official: sampling top-k 10, temperature 0.4, top-p 0.9, 50 new tokens; the answer is the first line,
            lower-cased, "(a)" / "(b)" / "answer:" removed (get_hf_answers' post-processing)
  scoring   task_eval/evaluation.eval_question_answering (F1 with stemming; multi-hop split by commas; open-domain
            answer up to ";")
Deviations (all forced or stated): Qwen3 is not in the official model list (MAX_LENGTH set to 32768; no conversation
needs truncation, the longest is 25.6k tokens); category 5 (adversarial) is left out because the released data have
no "answer" field for 444 of its 446 questions (the official code would fail; follow-up work reports categories 1-4);
thinking on (not in the official code) uses Qwen3's thinking sampling and 1024 + 50 new tokens, scored on the reply
after </think> (an unfinished thought gives an empty answer); generation runs on vLLM; one seed per question, shared
by all models (paired). --order chrono (secondary) changes one thing: sessions oldest first.
usage: python explore_locomo_official.py run --models Qwen3-4B [--think] [--order official|chrono] | stats
"""
import argparse
import glob
import json
import os
import random
import re
import sys
import types
import zlib

import numpy as np
import pandas as pd

OFFICIAL = "/data/locomo_official"
sys.path.insert(0, OFFICIAL)
sys.modules.setdefault("bert_score", types.SimpleNamespace(score=None))   # imported by evaluation.py, not used for F1
from task_eval.evaluation import eval_question_answering   # noqa: E402

_src = open(os.path.join(OFFICIAL, "task_eval/hf_llm_utils.py"), encoding="utf-8").read()
_ns = {"__file__": os.path.join(OFFICIAL, "task_eval/hf_llm_utils.py")}
exec(_src.split("def run_mistral")[0] + "\n" + _src[_src.index("def get_input_context"):_src.index("def get_hf_answers")], _ns)
get_input_context, QA_PROMPT, ANS = _ns["get_input_context"], _ns["QA_PROMPT"], _ns["ANS_TOKENS_PER_QUES"]
_ns["MAX_LENGTH"]["qwen3"] = 32768

from paths import data   # noqa: E402

DATA = data("locomo", "locomo10.json")
OUT = "results/locomo_official_{}.jsonl"
OFFICIAL_SAMPLING = dict(temperature=0.4, top_k=10, top_p=0.9)


def chrono_context(conv, tok, args):
    """secondary condition: the same text as get_input_context, sessions oldest first"""
    ctx = get_input_context(conv, "", tok, args)
    k = ctx.index("\nDATE: ")
    blocks = [b for b in re.split(r"(?=\nDATE: )", ctx[k:]) if b]
    return ctx[:k] + "".join(blocks[::-1])


def items(tok, order):
    a = argparse.Namespace(model="qwen3", batch_size=1)
    out = []
    for s in json.load(open(DATA, encoding="utf-8")):
        conv = s["conversation"]
        ctx = get_input_context(conv, "", tok, a) if order == "official" else chrono_context(conv, tok, a)
        for i, qa in enumerate(s["qa"]):
            if qa["category"] == 5:
                continue
            q = qa["question"] + (" Use DATE of CONVERSATION to answer with an approximate date." if qa["category"] == 2 else "")
            out.append(dict(sample_id=s["sample_id"], i=i, category=qa["category"], question=qa["question"],
                            answer=qa["answer"], user=ctx + "\n\n" + QA_PROMPT.format(q)))
    return out


def post(answer):   # get_hf_answers' post-processing for categories 1-4
    answer = answer.replace('\\"', "'").strip()
    lines = [w.strip() for w in answer.split("\n") if not w.strip().isspace()]
    answer = lines[0] if lines else ""
    return answer.lower().replace("(a)", "").replace("(b)", "").replace("a)", "").replace("b)", "").replace("answer:", "").strip()


def run(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from transformers import AutoTokenizer
    from app_common import JsonlAppender, MODELS, out_tag
    from run_app_fix import QWEN_THINK_SAMPLING, stop_ids
    from vllm_client import VLLMReader
    for name in args.models.split(","):
        hf = MODELS.get(name, (name, False))[0]
        tok = AutoTokenizer.from_pretrained(hf)
        its = items(tok, args.order)
        w = JsonlAppender(OUT.format(out_tag(name, args.think) + ("" if args.order == "official" else "~chrono")),
                          key=lambda r: (r["sample_id"], r["i"]))
        todo = [x for x in its if (x["sample_id"], x["i"]) not in w.done]
        print(name, args.order, "think" if args.think else "", "todo", len(todo), "of", len(its), flush=True)
        if not todo:
            w.close()
            continue
        model = VLLMReader(hf)
        stop = stop_ids(tok)
        if args.think:
            samp = {k: v for k, v in QWEN_THINK_SAMPLING.items() if k != "do_sample"}
            budget = 1024 + ANS
        else:
            samp, budget = OFFICIAL_SAMPLING, ANS

        def one(x):
            p = tok.apply_chat_template([{"role": "user", "content": x["user"]}], tokenize=False, add_generation_prompt=True,
                                        enable_thinking=args.think)
            ids = tok(p, add_special_tokens=False).input_ids
            seed = zlib.crc32(str(("locomo-official", x["sample_id"], x["i"])).encode())
            return x, len(ids), model.sample(ids, budget, stop, seed, **samp)

        with ThreadPoolExecutor(32) as ex:
            futs = [ex.submit(one, x) for x in todo]
            for n, fut in enumerate(as_completed(futs)):
                x, n_tok, out_ids = fut.result()
                text = tok.decode(out_ids, skip_special_tokens=True)
                fin = "</think>" in text if args.think else True
                vis = text.split("</think>", 1)[1] if (args.think and fin) else ("" if args.think else text)
                pred = post(vis)
                f1 = eval_question_answering([dict(category=x["category"], answer=x["answer"], prediction=pred)], "prediction")[0][0]
                w.write({k: v for k, v in x.items() if k != "user"} | dict(n_tok=n_tok, finished=fin, raw=text, prediction=pred,
                                                                        f1=float(f1)))
                if n % 300 == 0:
                    print(n, "/", len(todo), flush=True)
        w.close()


def stats(args):
    CAT = {1: "multi-hop", 2: "temporal", 3: "open-domain", 4: "single-hop"}
    files = sorted(f for f in glob.glob(OUT.format("*")))
    dfs = {os.path.basename(f)[len("locomo_official_"):-len(".jsonl")]: pd.DataFrame([json.loads(l) for l in open(f)]) for f in files}
    for t, d in dfs.items():
        by = d.groupby("category").f1.mean().mul(100).round(1)
        print(f"{t:45s} n={len(d):4d} | F1 all {d.f1.mean() * 100:5.1f} | " + " ".join(f"{CAT[c]} {v}" for c, v in by.items()) +
              (f" | unfinished {100 - d.finished.mean() * 100:.1f}%" if "+think" in t else ""))
    pairs = [(a, b) for a in dfs for b in dfs if a != b and args.pairs and f"{a}|{b}" in args.pairs.split(",")]
    for a, b in pairs:
        m = dfs[a].merge(dfs[b], on=["sample_id", "i"], suffixes=("_a", "_b"))
        rng = np.random.default_rng(0)
        for name, g in [("all", m)] + [(f"cat{c}", m[m.category_a == c]) for c in (1, 2, 3, 4)]:
            d = (g.f1_a - g.f1_b).values
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1) * 100
            convs = g.sample_id.unique()
            grp = {c: (g[g.sample_id == c].f1_a - g[g.sample_id == c].f1_b).values for c in convs}
            cl = [np.concatenate([grp[c] for c in rng.choice(convs, len(convs))]).mean() * 100 for _ in range(4000)]
            print(f"  {a} - {b} {name:5s} {d.mean() * 100:+5.1f} [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]"
                  f" cluster(conv) [{np.percentile(cl, 2.5):+.1f}, {np.percentile(cl, 97.5):+.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "stats", "check"])
    ap.add_argument("--models", default="Qwen3-4B")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--order", default="official", choices=["official", "chrono"])
    ap.add_argument("--pairs", default="")
    args = ap.parse_args()
    if args.stage == "check":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        for order in ("official", "chrono"):
            its = items(tok, order)
            x = its[0]["user"]
            dates = re.findall(r"DATE: (.*)", x)
            print(order, len(its), "questions | first 2 dates", dates[:2], "| last", dates[-1], "| blocks", len(dates))
        a = argparse.Namespace(model="qwen3", batch_size=1)
        ok = True
        for smp in json.load(open(DATA, encoding="utf-8")):
            o, c = get_input_context(smp["conversation"], "", tok, a), chrono_context(smp["conversation"], tok, a)
            k = o.index("\nDATE: ")
            bo, bc = ([b for b in re.split(r"(?=\nDATE: )", t[k:]) if b] for t in (o, c))
            ok &= o[:k] == c[:k] and bo == bc[::-1]
        print("chrono = official blocks reversed, same text:", ok)
        return
    {"run": run, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()
