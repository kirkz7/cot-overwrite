"""E18.2 (CLOUD_NOTEBOOK.md "E18.2"): run a served model (vLLM, thinking on, Qwen3 sampling, one seed per item, budget
1024 - the same settings as the PersonaMem test) on the synthetic many-record prompts (gen_bind_data_v5.py many).
  harvest : data_train/many_harvest.jsonl -> its looping thoughts become unlikelihood rows (data_train/ul_loops<tag>.jsonl)
  eval    : data_train/many_dev.jsonl (held-out genres) -> unfinished / looping / correct rates (the in-distribution
            loop check; synthetic data, not a test set)
Looping = the thought did not finish and > 30% of its non-empty lines repeat an earlier line (same rule as the
PersonaMem diagnostic). Unlikelihood spans = every line that repeats an earlier line of the same thought.
usage: COT_VLLM_URL=... python cloud/harvest_loops.py harvest|eval <served model path> <tag>
"""
import json
import os
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor, as_completed

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
from transformers import AutoTokenizer

from app_common import JsonlAppender, chat_prompt, load_jsonl, strip_think
from run_app_fix import QWEN_THINK_SAMPLING, stop_ids
from train_lora import norm
from vllm_client import VLLMReader

mode, served, tag = sys.argv[1], sys.argv[2], sys.argv[3]
src = {"harvest": "data_train/many_harvest.jsonl", "eval": "data_train/many_dev.jsonl"}[mode]
out = f"results/many_{mode}_{tag}.jsonl"
tok = AutoTokenizer.from_pretrained(served)
model = VLLMReader(served)
kw = {k: v for k, v in QWEN_THINK_SAMPLING.items() if k != "do_sample"}


def repeat_spans(text):
    """char spans of lines (with their newline) that repeat an earlier non-empty line"""
    seen, spans, pos = set(), [], 0
    for line in text.splitlines(keepends=True):
        k = line.strip()
        if k and k in seen:
            spans.append([pos, pos + len(line)])
        seen.add(k)
        pos += len(line)
    return spans


STOP = stop_ids(tok)


def one(x, ids):   # worker threads only send requests; tokenising and decoding stay in the main thread
    return x, model.sample(ids, 1024, STOP, zlib.crc32(str(("many", x["id"])).encode()), **kw)


def score(x, out_ids):
    gen = tok.decode(out_ids, skip_special_tokens=True)
    fin = "</think>" in gen
    th = gen.split("</think>")[0]
    lines = [l.strip() for l in th.splitlines() if l.strip()]
    rep = 1 - len(set(lines)) / max(1, len(lines))
    vis = strip_think(gen).strip()
    return dict(id=x["id"], qtype=x["qtype"], k=x["k"], finished=fin, looping=(not fin) and rep > 0.3, rep=round(rep, 3),
                correct=fin and all(norm(n) in norm(vis) for n in x["need"]), gen=gen)


rows = load_jsonl(src)
w = JsonlAppender(out, key=lambda r: r["id"])
todo = [x for x in rows if x["id"] not in w.done]
# 32 requests in flight (vLLM batches them); each request carries its own seed, so the order does not matter
with ThreadPoolExecutor(32) as ex:
    futs = [ex.submit(one, x, tok(chat_prompt(tok, x["prompt"], think=True), add_special_tokens=False).input_ids)
            for x in todo]
    for i, fut in enumerate(as_completed(futs)):
        w.write(score(*fut.result()))
        if i % 100 == 0:
            print(i, "/", len(todo), flush=True)
w.close()

d = pd.DataFrame(load_jsonl(out))
print(f"{mode} {tag}: n {len(d)} | unfinished {100 - d.finished.mean() * 100:.1f}% | looping {d.looping.mean() * 100:.1f}% | "
      f"correct {d.correct.mean() * 100:.1f}%")
print("by qtype:", d.groupby("qtype")[["finished", "looping", "correct"]].mean().mul(100).round(1).to_dict("index"))
print("by records:", d.groupby(d.k + 1)[["looping", "correct"]].mean().mul(100).round(1).to_dict("index"))
if mode == "harvest":
    src_by_id = {x["id"]: x for x in rows}
    ul = [dict(id=f"ul-{r.id}", kind="ul", ul=True, think=True, prompt=src_by_id[r.id]["prompt"], answer=r.gen,
               ul_spans=repeat_spans(r.gen), qtype="ul", dated=True, order=src_by_id[r.id]["order"])
          for r in d[d.looping].itertuples()]
    path = f"data_train/ul_loops{tag}.jsonl"
    with open(path, "w", encoding="utf-8") as f:
        for r in ul:
            f.write(json.dumps(r) + "\n")
    print(path, len(ul), "rows | penalised characters share",
          round(sum(e - s for r in ul for s, e in r["ul_spans"]) / max(1, sum(len(r["answer"]) for r in ul)), 3))
