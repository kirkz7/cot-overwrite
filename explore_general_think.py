"""P6b (CLOUD_NOTEBOOK.md pre-registration, 10-07, user): the no-harm suite with Qwen3 THINKING ON, base Qwen3-4B vs E18.
explore_general.py (thinking off) is left untouched; this file copies its tasks, limits and harness and changes only
what thinking needs:
  - IFEval (all 541) and LongBench-E qasper / multifieldqa_en / hotpotqa / 2wikimqa (first 50 each) through lm-eval with
    enable_thinking=True; scored on the visible reply (lm-eval think_end_token="</think>"; an unfinished thought leaves
    no reply). max_length = 16384 + the generation budget, so LongBench keeps the same 16k-token context as thinking off.
  - GSM8K (the same 250 problems and scoring as run_ext_eval.py) on the visible reply (app_common.strip_think).
  - decoding: Qwen3's recommended thinking sampling (temperature 0.6, top-p 0.95, top-k 20; desktop E18 protocol 10-06),
    budget 4096 new tokens (thought + reply). Seeds fixed: lm-eval seeds 0/1234/1234; GSM8K seeded per batch.
  - batch as in explore_general: IFEval 4, LongBench 1; GSM8K 8 (left-padded batches).
MMLU / ARC-C / HellaSwag are log-likelihood tasks (no generation, no chat template): thinking does not apply; P6 covers them.
usage: python explore_general_think.py run --models Qwen3-4B | stats --models Qwen3-4B,Qwen3-4B@runs/e18-dec/final
"""
import argparse
import json
import math
import os
import zlib

from explore_general import GEN_LIMIT, MAX_LEN, METRIC

OUT = "results/general_think_{}.json"
GEN_TASKS = ["ifeval", "longbench_qasper_e", "longbench_multifieldqa_en_e", "longbench_hotpotqa_e", "longbench_2wikimqa_e"]
BUDGET = 4096
SAMPLING = "do_sample=True,temperature=0.6,top_p=0.95,top_k=20,max_gen_toks=%d" % BUDGET


def gsm8k(tok, model, done, path):
    import torch
    from tqdm import tqdm
    from app_common import chat_prompt, strip_think
    from run_app_fix import QWEN_THINK_SAMPLING, stop_ids
    from run_ext_eval import gsm8k_items
    import re
    def score(it, text):   # run_ext_eval.run_gsm8k's rule, on the visible reply
        m = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text.replace("**", ""))
        nums = m or re.findall(r"-?\d[\d,]*\.?\d*", text)
        try:
            pred = float(nums[-1].replace(",", "")) if nums else None
        except ValueError:
            pred = None
        return pred is not None and abs(pred - it["gold"]) < 1e-6
    rows = done.get("gsm8k_rows", [])
    seen = {r["i"] for r in rows}
    items = [it for it in gsm8k_items() if it["i"] not in seen]
    tok.padding_side = "left"
    pad = tok.pad_token_id if tok.pad_token_id is not None else tok.eos_token_id
    for b in tqdm(range(0, len(items), 8), desc="gsm8k think"):
        batch = items[b:b + 8]
        enc = tok([chat_prompt(tok, it["user"], think=True) for it in batch], return_tensors="pt", padding=True,
                  add_special_tokens=False).to("cuda")
        torch.manual_seed(zlib.crc32(str(batch[0]["i"]).encode()))
        out = model.generate(**enc, max_new_tokens=BUDGET, eos_token_id=sorted(stop_ids(tok)), pad_token_id=pad,
                             **QWEN_THINK_SAMPLING)
        for it, o in zip(batch, out[:, enc.input_ids.shape[1]:]):
            text = tok.decode(o, skip_special_tokens=True)
            vis = strip_think(text)
            rows.append(dict(i=it["i"], correct=score(it, vis), unfinished="</think>" not in text, n_new=int((o != pad).sum())))
        done["gsm8k_rows"] = rows
        json.dump(done, open(path, "w", encoding="utf-8"), indent=1)
    acc = sum(r["correct"] for r in rows) / len(rows)
    done["gsm8k"] = {"gsm8k": {"acc,none": acc, "acc_stderr,none": math.sqrt(acc * (1 - acc) / len(rows)),
                               "unfinished": sum(r["unfinished"] for r in rows) / len(rows), "n": len(rows)}}
    json.dump(done, open(path, "w", encoding="utf-8"), indent=1)


def run(args):
    import torch
    from lm_eval.models.huggingface import HFLM
    from app_common import tag, load_reader, free_gpu
    for name in args.models.split(","):
        path = OUT.format(tag(name))
        done = json.load(open(path, encoding="utf-8")) if os.path.exists(path) else {}
        todo = [t for t in GEN_TASKS if t not in done]
        print(name, "todo", todo + ([] if "gsm8k" in done else ["gsm8k"]), flush=True)
        if not todo and "gsm8k" in done:
            continue
        tok, model = load_reader(name)
        lm = HFLM(pretrained=model, tokenizer=tok, batch_size=4, max_length=MAX_LEN + BUDGET, enable_thinking=True,
                  think_end_token="</think>")
        for t in todo:
            lm.batch_size_per_gpu = 1 if t.startswith("longbench") else 4
            from lm_eval import simple_evaluate
            r = simple_evaluate(model=lm, tasks=[t], limit=GEN_LIMIT.get(t), apply_chat_template=True, log_samples=False,
                                bootstrap_iters=1000, gen_kwargs=SAMPLING, random_seed=0, numpy_random_seed=1234,
                                torch_random_seed=1234)
            done[t] = dict(r["results"])
            print(t, {k: v for k, v in r["results"].get(t, {}).items() if not k.startswith("alias")}, flush=True)
            json.dump(done, open(path, "w", encoding="utf-8"), indent=1)
            torch.cuda.empty_cache()
        if "gsm8k" not in done:
            gsm8k(tok, model, done, path)
        del lm, tok, model
        free_gpu()


def stats(args):
    from app_common import tag
    rows = {}
    for name in args.models.split(","):
        path = OUT.format(tag(name))
        if os.path.exists(path):
            r = json.load(open(path, encoding="utf-8"))
            rows[tag(name)] = {t: (r.get(t, {}).get(t, {}).get(m), r.get(t, {}).get(t, {}).get(m.replace(",", "_stderr,")))
                               for t, m in list({k: METRIC[k] for k in GEN_TASKS}.items()) + [("gsm8k", "acc,none")]}
    print(json.dumps(rows, indent=1))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "stats"])
    ap.add_argument("--models", default="Qwen3-4B")
    args = ap.parse_args()
    {"run": run, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()
