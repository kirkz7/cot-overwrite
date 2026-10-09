"""Option C no-harm check (user 10-08 09:40; CLOUD_NOTEBOOK "方案 C"): does presence_penalty hurt general generation?
vLLM, thinking on, Qwen3 sampling (temperature 0.6, top-p 0.95, top-k 20), budget 4096 (as P6b), one seed per item shared
by both penalties (paired). Tasks: IFEval (all 541; prompt as the user message, scored by lm-eval's ifeval
process_results on the visible reply) and GSM8K (the same 250 problems and rule as run_ext_eval / explore_general_think).
An unfinished thought leaves an empty reply (scored as is; the unfinished share is reported).
usage: COT_VLLM_URL=... python cloud/general_pp.py <served model path> <tag> [penalties, default 0,1.5]
"""
import os
import re
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import datasets
import pandas as pd
from transformers import AutoTokenizer

from app_common import JsonlAppender, chat_prompt, load_jsonl
from run_app_fix import QWEN_THINK_SAMPLING, stop_ids
from run_ext_eval import gsm8k_items
from vllm_client import VLLMReader
from lm_eval.tasks.ifeval.utils import process_results

served, tag = sys.argv[1], sys.argv[2]
pens = [float(x) for x in (sys.argv[3] if len(sys.argv) > 3 else "0,1.5").split(",")]
out = f"results/general_pp_{tag}.jsonl"
tok = AutoTokenizer.from_pretrained(served)
model = VLLMReader(served)
STOP = stop_ids(tok)
kw = {k: v for k, v in QWEN_THINK_SAMPLING.items() if k != "do_sample"}


def gsm_ok(gold, text):   # explore_general_think.gsm8k's rule (= run_ext_eval.run_gsm8k), on the visible reply
    m = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text.replace("**", ""))
    nums = m or re.findall(r"-?\d[\d,]*\.?\d*", text)
    try:
        pred = float(nums[-1].replace(",", "")) if nums else None
    except ValueError:
        pred = None
    return pred is not None and abs(pred - gold) < 1e-6


items = [dict(task="ifeval", i=i, user=d["prompt"], doc=d) for i, d in enumerate(datasets.load_dataset("google/IFEval", split="train"))]
items += [dict(task="gsm8k", i=it["i"], user=it["user"], gold=it["gold"]) for it in gsm8k_items()]
w = JsonlAppender(out, key=lambda r: (r["task"], r["i"], r["pp"]))
todo = [(x, p) for p in pens for x in items if (x["task"], x["i"], p) not in w.done]
print(tag, "todo", len(todo), flush=True)


def one(x, p, ids):
    extra = {"presence_penalty": p} if p else {}
    return x, p, model.sample(ids, 4096, STOP, zlib.crc32(str(("gpp", x["task"], x["i"])).encode()), **kw, **extra)


with ThreadPoolExecutor(32) as ex:
    futs = [ex.submit(one, x, p, tok(chat_prompt(tok, x["user"], think=True), add_special_tokens=False).input_ids) for x, p in todo]
    for n, fut in enumerate(as_completed(futs)):
        x, p, ids = fut.result()
        text = tok.decode(ids, skip_special_tokens=True)
        fin = "</think>" in text
        vis = text.split("</think>", 1)[1].strip() if fin else ""
        row = dict(task=x["task"], i=x["i"], pp=p, finished=fin, n_new=len(ids))
        if x["task"] == "ifeval":
            r = process_results(x["doc"], [vis])
            row.update(strict=bool(r["prompt_level_strict_acc"]), loose=bool(r["prompt_level_loose_acc"]),
                       inst_strict=[bool(v) for v in r["inst_level_strict_acc"]])
        else:
            row.update(correct=gsm_ok(x["gold"], vis))
        w.write(row)
        if n % 200 == 0:
            print(n, "/", len(todo), flush=True)
w.close()

d = pd.DataFrame(load_jsonl(out))
for p, g in d.groupby("pp"):
    f, gs = g[g.task == "ifeval"], g[g.task == "gsm8k"]
    inst = [v for l in f.inst_strict for v in l]
    print(f"{tag} pp={p}: IFEval prompt-strict {f.strict.mean() * 100:.1f} loose {f.loose.mean() * 100:.1f} "
          f"inst-strict {sum(inst) / max(1, len(inst)) * 100:.1f} unfinished {100 - f.finished.mean() * 100:.1f}% | "
          f"GSM8K {gs.correct.mean() * 100:.1f} unfinished {100 - gs.finished.mean() * 100:.1f}%")
