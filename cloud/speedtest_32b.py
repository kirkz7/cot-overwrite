"""vLLM decision (CLOUD_NOTEBOOK.md "vLLM"): time the same 10 long MemConflict prompts on Qwen3-32B with HF (two-card
pipeline, COT_DEVICE_MAP=auto) and with vLLM (two-card tensor parallel, COT_ENGINE=vllm + cloud/vllm_ctl.sh up).
Nothing is written to results/; per-item times and outputs go to logs/speed_<engine>.json.
usage: COT_DEVICE_MAP=auto python cloud/speedtest_32b.py hf
       COT_ENGINE=vllm     python cloud/speedtest_32b.py vllm
       python cloud/speedtest_32b.py compare
"""
import json
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))


def pick():
    """10 chrono MemConflict prompts spread evenly over the length range (same choice for both engines)"""
    import explore_extmem as e
    its = sorted([x for x in e.memconf() if x["cond"] == "chrono"], key=lambda x: len(x["user"]))
    return [its[round(i * (len(its) - 1) / 9)] for i in range(10)]


def run(engine):
    import torch
    from app_common import chat_prompt, load_reader
    from explore_extmem import BUDGET
    from explore_probe import LONG_SDPA, sdpa_kernel
    from run_app_fix import generate
    its = pick()
    tok, model = load_reader("Qwen3-32B")
    rows = []
    for x in its:
        p = chat_prompt(tok, x["user"])
        n = len(tok(p, add_special_tokens=False).input_ids)
        torch.cuda.synchronize()
        t = time.time()
        with sdpa_kernel(LONG_SDPA, set_priority=True):
            text = generate(tok, model, p, BUDGET["memconf"])
        torch.cuda.synchronize()
        rows.append(dict(id=x["id"], n_tok=n, sec=time.time() - t, text=text))
        print(f"{engine} {x['id']:>12} {n:6d} tok {rows[-1]['sec']:6.1f} s", flush=True)
    json.dump(rows, open(f"logs/speed_{engine}.json", "w"), indent=1)


def compare():
    h, v = (json.load(open(f"logs/speed_{e}.json")) for e in ("hf", "vllm"))
    th, tv = sum(r["sec"] for r in h), sum(r["sec"] for r in v)
    for a, b in zip(h, v):
        print(f"{a['n_tok']:6d} tok  hf {a['sec']:6.1f} s  vllm {b['sec']:6.1f} s  same text: {a['text'] == b['text']}")
    print(f"total hf {th:.0f} s, vllm {tv:.0f} s, speed-up {th / tv:.2f}x (decision rule: >= 1.5)")
    print(f"same text {sum(a['text'] == b['text'] for a, b in zip(h, v))}/10")


if __name__ == "__main__":
    compare() if sys.argv[1] == "compare" else run(sys.argv[1])
