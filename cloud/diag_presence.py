"""Diagnostic only (10-08, user): do E18.1-8B's unfinished PersonaMem thoughts (thinking on, budget 1024; most are
looping record lists) finish when the Qwen3 model card's anti-repetition setting presence_penalty=1.5 is added?
Same prompts, seeds, sampling (temperature 0.6, top-p 0.95, top-k 20) and budget as the official run; only
presence_penalty changes. Not a protocol change: the official results stay as run. Aggregates only (PersonaMem is
held-out, not blind for E18 / E18.1).
usage: COT_VLLM_URL=http://127.0.0.1:8002 python cloud/diag_presence.py [penalty]"""
import os
import re
import sys
import zlib

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
import pandas as pd
from transformers import AutoTokenizer

from app_common import chat_prompt, load_jsonl, strip_think
from explore_longconv import personamem, pm_pred
from run_app_fix import QWEN_THINK_SAMPLING, stop_ids
from vllm_client import VLLMReader

MERGED = "/root/cot-overwrite/runs/e181-q8/merged"
pen = float(sys.argv[1]) if len(sys.argv) > 1 else 1.5
d = pd.DataFrame(load_jsonl("results/longconv_Qwen3-8B-e181~vllm+think.jsonl"))
d = d[d.task == "personamem"]
unf = d[~d.full.astype(str).str.contains("</think>")]
keys = set(zip(unf.id, unf.cond))
its = {(x["id"], x["cond"]): x for x in personamem() if (x["id"], x["cond"]) in keys}
tok = AutoTokenizer.from_pretrained(MERGED)
model = VLLMReader(MERGED)
kw = {k: v for k, v in QWEN_THINK_SAMPLING.items() if k != "do_sample"}


def loops(t):
    ls = [l.strip() for l in t.splitlines() if l.strip()]
    return 1 - len(set(ls)) / max(1, len(ls)) > 0.3


rows = []
for (iid, cond), x in sorted(its.items()):
    ids = tok(chat_prompt(tok, x["user"], think=True), add_special_tokens=False).input_ids
    out = tok.decode(model.sample(ids, 1024, stop_ids(tok), zlib.crc32(str((x["task"], x["id"], x["cond"])).encode()),
                                  **kw, presence_penalty=pen), skip_special_tokens=True).strip()
    fin = "</think>" in out
    rows.append(dict(id=iid, cond=cond, qtype=x["qtype"], finished=fin, looping=(not fin) and loops(out.split("</think>")[0]),
                     correct=pm_pred(out) == x["gold"]))
r = pd.DataFrame(rows)
print(f"presence_penalty={pen}: {len(r)} previously unfinished items | now finished {r.finished.mean() * 100:.1f}% | "
      f"still looping {r.looping.mean() * 100:.1f}% | correct {r.correct.mean() * 100:.1f}%")
print("by qtype:", r.groupby(r.qtype.str.slice(0, 12))[["finished", "correct"]].mean().mul(100).round(1).to_dict("index"))
r.to_json(f"results/diag_presence_{pen}.jsonl", orient="records", lines=True)
