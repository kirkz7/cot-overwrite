"""Concurrent vLLM runner for the explore_extmem items (CLOUD_PLAN P8, lease end 10-09). Same items, prompts, decoding,
per-item seeds, answer parsing and record format as `explore_extmem.py run`, and the same results file, so
`explore_extmem.py judge / stats` and the resume rules work unchanged. It adds only:
  - concurrent requests to the vLLM server (the server batches them; the original loop sends one at a time),
  - --conds: run only some presentation orders (e.g. chrono,rev),
  - a fixed pseudo-random item order (crc32 of the item id; every requested order of an item is submitted together), so a
    run stopped at a deadline leaves a random subsample whose finished items have all their orders,
  - --limit: only the first n jobs of that order, and --out: another results file (smoke tests that must not touch
    the real results).
Rows are written in submission order from the main thread. Cloud only (needs COT_ENGINE=vllm and a running server).
usage: COT_ENGINE=vllm COT_VLLM_MODEL=Qwen/Qwen3-32B COT_VLLM_TP=2 \
       python cloud/run_extmem_conc.py --models Qwen3-32B --tasks memconf --conds chrono,rev --think --budget 1024 --conc 8
"""
import argparse
import os
import sys
import zlib
from concurrent.futures import ThreadPoolExecutor

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import explore_extmem as em  # noqa: E402
from app_common import JsonlAppender, chat_prompt, load_reader, out_tag, strip_think  # noqa: E402
from run_app_fix import generate_think, stop_ids  # noqa: E402


def record(x, text, n_tok):
    """the row explore_extmem.run writes for item x and model output text"""
    vis = strip_think(text)
    resp = vis.rsplit("Answer:", 1)[1].strip().split("\n")[0] if "Answer:" in vis else vis.strip().split("\n")[0]
    rec = {k: v for k, v in x.items() if k != "user"}
    rec.update(n_tok=n_tok, response=resp, full=text)
    if x["task"] in ("mabcr", "mabcr32k"):
        hit = any(em.norm(g) in em.norm(resp) for g in x["gold"])
        rec.update(correct=hit, stale=bool(x["old"]) and not hit and em.norm(x["old"]) in em.norm(resp))
    return rec


def ordered(todo, conds):
    return sorted(todo, key=lambda x: (zlib.crc32(str(x["id"]).encode()), x["task"], conds.index(x["cond"])))


def run(args):
    conds = args.conds.split(",")
    assert set(conds) <= set(em.CONDS), f"--conds must be a subset of {em.CONDS}"
    tasks = args.tasks.split(",")
    its = [x for x in em.items(args.tasks) if x["task"] in tasks and x["cond"] in conds]
    for name in args.models.split(","):
        out = args.out or em.OUT.format(out_tag(name, args.think))
        w = JsonlAppender(out, key=lambda r: (r["task"], r["id"], r["cond"]))
        todo = ordered([x for x in its if (x["task"], x["id"], x["cond"]) not in w.done], conds)
        if args.limit:
            todo = todo[: args.limit]
        print(name, "->", out, "todo", len(todo), "conc", args.conc, flush=True)
        if not todo:
            w.close()
            continue
        tok, model = load_reader(name)
        assert getattr(model, "is_vllm", False), "this runner is for COT_ENGINE=vllm only (use explore_extmem.py run on HF)"
        stop = stop_ids(tok)

        def one(x):
            p = chat_prompt(tok, x["user"], think=args.think)
            ids = tok(p, add_special_tokens=False).input_ids
            if args.think:   # Qwen3 recommended sampling, seeded per item (run_app_fix.generate_think)
                text = generate_think(tok, model, p, args.budget or 1024, (x["task"], x["id"], x["cond"]))
            else:            # = run_app_fix.generate on the vLLM path (probe.greedy forwards the same ids)
                text = tok.decode(model.greedy(ids, args.budget or em.BUDGET[x["task"]], stop), skip_special_tokens=True).strip()
            return record(x, text, len(ids))

        with ThreadPoolExecutor(args.conc) as ex:
            for k, rec in enumerate(ex.map(one, todo), 1):
                w.write(rec)
                if k % 20 == 0 or k == len(todo):
                    print(k, "/", len(todo), flush=True)
        w.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", required=True)
    ap.add_argument("--tasks", default="memconf")
    ap.add_argument("--conds", default="chrono,rev,retr")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--budget", type=int, default=None)
    ap.add_argument("--conc", type=int, default=int(os.environ.get("COT_VLLM_CONC", "8")))
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", default="", help="results file (default: the one explore_extmem.py run writes)")
    run(ap.parse_args())


if __name__ == "__main__":
    main()
