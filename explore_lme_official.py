"""LongMemEval end to end with the official retrieval + reading code, with and without its date sort
(user 10-08 10:10; CLOUD_NOTEBOOK "LongMemEval 官方检索 + 作答", written before running). LongMemEval is SEEN data
(used in design): flag it whenever it is used as evidence. Never print conversation text (aggregates only).
Official code: xiaowu0162/LongMemEval commit 9e0b455f (copied to /data/lme_official):
  retrieval  src/retrieval/run_retrieval.py, flat-bm25 at session granularity (rank_bm25.BM25Okapi over the user
             turns of each session, whitespace tokens) - re-implemented from the two official functions below
  reading    src/generation/run_generation.prepare_prompt (retriever_type flat-session, history_format json,
             useronly false, cot true = run_generation.sh defaults), greedy, gen_length 800, the history truncated
             to model_max_length - gen_length - 1000 tokens (as the official script)
  judging    src/evaluation/evaluate_qa.get_anscheck_prompt (type-specific prompts; "_abs" questions use the
             abstention prompt); label = "yes" in the reply
The one change under test (--order relevance): prepare_prompt's `retrieved_chunks.sort(key=lambda x: x[0])` (sort
sessions by date) is switched off, so the top-k sessions stay in BM25 rank order, as in most RAG / memory systems.
Deviations: Qwen3 is not in the official model table (max length 32768); top-k 5 sessions (the shell default is 50;
top-10 already truncated 22% of the histories, and truncation keeps the start of the history - the OLDEST sessions after
the date sort - so the two orders would lose different sessions; top-5 truncates none); generation on vLLM; thinking on (not in the official code) uses Qwen3's thinking sampling and
1024 + 800 new tokens, scored on the reply after </think>; the judge is Qwen3-14B 4-bit with the official prompts
(the official judge is GPT-4o or a local Llama-3.1-70B).
usage: python explore_lme_official.py run --models Qwen3-4B [--think] [--order date|relevance] | judge | stats
"""
import argparse
import contextlib
import copy
import glob
import io
import json
import os
import sys
import zlib

import numpy as np
import pandas as pd

LME = "/data/lme_official"
S_FILE = None   # resolved in main (paths.hub)
OUT = "results/lme_official_{}.jsonl"
TOPK, GEN, MAXLEN = 5, 800, 32768   # top-5: no history is truncated (top-10 truncated 22%, see the notebook)


def official(path, start, end, patch=None):
    src = open(os.path.join(LME, path), encoding="utf-8").read()
    code = src[src.index(start):src.index(end)] if end else src[src.index(start):]
    for a, b in (patch or []):
        assert code.count(a) == 1, a
        code = code.replace(a, b)
    return code


_ns = {"json": json, "np": np}
# the cleaned data omit has_answer on turns without the answer; it only renames an answer session without evidence to
# "noans_" (affects the recall metric, not the ranking or the prompt)
exec(official("src/retrieval/run_retrieval.py", "def process_item_flat_index", "def batch_get_retrieved_context_and_eval",
              patch=[("all([not turn['has_answer'] for", "all([not turn.get('has_answer', False) for")]), _ns)
exec(official("src/generation/run_generation.py", "def prepare_prompt", "@backoff.on_exception",
              patch=[("    retrieved_chunks.sort(key=lambda x: x[0])", "    if SORT_BY_DATE:\n        retrieved_chunks.sort(key=lambda x: x[0])")]), _ns)
exec(official("src/evaluation/evaluate_qa.py", "def get_anscheck_prompt", "if __name__ == '__main__':"), _ns)
process_item_flat_index, prepare_prompt, get_anscheck_prompt = _ns["process_item_flat_index"], _ns["prepare_prompt"], _ns["get_anscheck_prompt"]


def with_retrieval(entry):
    """run_retrieval.batch_get_retrieved_context_and_eval, flat-bm25 / session: adds entry['retrieval_results']"""
    from rank_bm25 import BM25Okapi
    corpus, ids, ts = [], [], []
    for sid, sess, d in zip(entry["haystack_session_ids"], entry["haystack_sessions"], entry["haystack_dates"]):
        c, i, t = process_item_flat_index(sess, "session", sid, d)
        corpus += c; ids += i; ts += t
    scores = BM25Okapi([doc.split(" ") for doc in corpus]).get_scores(entry["question"].split(" "))
    ranks = np.argsort(scores)[::-1]
    correct = {c for c in ids if "answer" in c}
    top = [ids[r] for r in ranks[:TOPK]]
    entry = dict(entry, retrieval_results=dict(query=entry["question"], ranked_items=[
        dict(corpus_id=ids[r], text=corpus[r], timestamp=ts[r]) for r in ranks]))
    return entry, dict(recall_all=all(c in top for c in correct) if correct else None,
                       recall_any=any(c in top for c in correct) if correct else None)


def build(entry, tok, order, gen_len):
    _ns["SORT_BY_DATE"] = order == "date"
    e, rec = with_retrieval(copy.deepcopy(entry))   # the official prepare_prompt edits the turns in place (drops has_answer)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):      # the official function prints only "Truncating from ..." lines
        p = prepare_prompt(e, "flat-session", TOPK, False, "json", True, tokenizer=tok, tokenizer_backend="huggingface",
                           max_retrieval_length=MAXLEN - gen_len - 1000, merge_key_expansion_into_value="none")
    return p, rec, "Truncating" in buf.getvalue()


def load_s():
    from paths import hub
    return json.load(open(hub("datasets--xiaowu0162--longmemeval-cleaned", "snapshots", "98d7416c24c778c2fee6e6f3006e7a073259d48f",
                              "longmemeval_s_cleaned.json"), encoding="utf-8"))


def run(args):
    from concurrent.futures import ThreadPoolExecutor, as_completed
    from transformers import AutoTokenizer
    from app_common import JsonlAppender, MODELS, out_tag
    from run_app_fix import QWEN_THINK_SAMPLING, stop_ids
    from vllm_client import VLLMReader
    data = load_s()
    for name in args.models.split(","):
        hf = MODELS.get(name, (name, False))[0]
        tok = AutoTokenizer.from_pretrained(hf)
        w = JsonlAppender(OUT.format(out_tag(name, args.think) + f"~{args.order}"), key=lambda r: r["question_id"])
        todo = [e for e in data if e["question_id"] not in w.done]
        print(name, args.order, "think" if args.think else "", "todo", len(todo), flush=True)
        if not todo:
            w.close()
            continue
        model = VLLMReader(hf)
        stop = stop_ids(tok)
        samp = {k: v for k, v in QWEN_THINK_SAMPLING.items() if k != "do_sample"}
        budget = 1024 + GEN if args.think else GEN

        def one(e):
            p, rec, trunc = build(e, tok, args.order, GEN)
            chat = tok.apply_chat_template([{"role": "user", "content": p}], tokenize=False, add_generation_prompt=True,
                                           enable_thinking=args.think)
            ids = tok(chat, add_special_tokens=False).input_ids
            out = (model.sample(ids, budget, stop, zlib.crc32(str(("lme-official", e["question_id"])).encode()), **samp)
                   if args.think else model.greedy(ids, budget, stop))
            return e, rec, trunc, len(ids), out

        with ThreadPoolExecutor(16) as ex:
            futs = [ex.submit(one, e) for e in todo]
            for n, fut in enumerate(as_completed(futs)):
                e, rec, trunc, n_tok, out = fut.result()
                text = tok.decode(out, skip_special_tokens=True)
                fin = ("</think>" in text) if args.think else True
                hyp = (text.split("</think>", 1)[1] if fin else "").strip() if args.think else text.strip()
                w.write(dict(question_id=e["question_id"], question_type=e["question_type"], n_tok=n_tok, truncated=trunc,
                             finished=fin, hypothesis=hyp, **rec))
                if n % 100 == 0:
                    print(n, "/", len(todo), flush=True)
        w.close()


def judge(args):
    import torch
    from app_common import JsonlAppender, chat_prompt, load_jsonl, load_reader
    from probe import greedy
    ref = {e["question_id"]: e for e in json.load(open(glob.glob(
        "/data/hf_cache/hub/datasets--xiaowu0162--longmemeval-cleaned/snapshots/98d7416c24c778c2fee6e6f3006e7a073259d48f/longmemeval_oracle.json")[0], encoding="utf-8"))}
    tok = model = None
    for path in sorted(glob.glob(OUT.format("*"))):
        if path.endswith("_judged.jsonl"):
            continue
        w = JsonlAppender(path.replace(".jsonl", "_judged.jsonl"), key=lambda r: r["question_id"])
        todo = [r for r in load_jsonl(path) if r["question_id"] not in w.done]
        print(os.path.basename(path), "todo", len(todo), flush=True)
        if todo and model is None:
            tok, model = load_reader("Qwen3-14B")
            eos = {tok.convert_tokens_to_ids("<|im_end|>"), tok.eos_token_id}
        for r in todo:
            q = ref[r["question_id"]]
            p = get_anscheck_prompt(r["question_type"], q["question"], q["answer"], r["hypothesis"], abstention="_abs" in r["question_id"])
            ids = torch.tensor([tok(chat_prompt(tok, p), add_special_tokens=False).input_ids], device="cuda")
            with torch.no_grad():
                reply = tok.decode(greedy(model, ids, 10, eos), skip_special_tokens=True)
            w.write(r | dict(judge=reply.strip()[:20], correct="yes" in reply.lower()))
        w.close()


def stats(args):
    files = sorted(glob.glob(OUT.format("*").replace(".jsonl", "_judged.jsonl")))
    dfs = {os.path.basename(f)[len("lme_official_"):-len("_judged.jsonl")]: pd.DataFrame([json.loads(l) for l in open(f)]) for f in files}
    for t, d in dfs.items():
        by = d.groupby("question_type").correct.mean().mul(100).round(1).to_dict()
        print(f"{t:45s} n={len(d)} | all {d.correct.mean() * 100:5.1f} | KU {by.get('knowledge-update')} | TR {by.get('temporal-reasoning')} | "
              f"MS {by.get('multi-session')} | truncated {d.truncated.mean() * 100:.0f}% | unfinished {100 - d.finished.mean() * 100:.1f}% | "
              f"recall_all@{TOPK} {d.recall_all.dropna().mean() * 100:.0f}%")
    rng = np.random.default_rng(0)
    for a, b in [p.split("|") for p in args.pairs.split(",") if p]:
        m = dfs[a].merge(dfs[b], on="question_id", suffixes=("_a", "_b"))
        for name, g in [("all", m), ("KU", m[m.question_type_a == "knowledge-update"]), ("TR", m[m.question_type_a == "temporal-reasoning"])]:
            d = (g.correct_a.astype(float) - g.correct_b.astype(float)).values
            bs = d[rng.integers(0, len(d), (10000, len(d)))].mean(1) * 100
            print(f"  {a} - {b} {name:3s} n={len(d)} {d.mean() * 100:+5.1f} [{np.percentile(bs, 2.5):+.1f}, {np.percentile(bs, 97.5):+.1f}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "judge", "stats", "check"])
    ap.add_argument("--models", default="Qwen3-4B")
    ap.add_argument("--think", action="store_true")
    ap.add_argument("--order", default="date", choices=["date", "relevance"])
    ap.add_argument("--pairs", default="")
    args = ap.parse_args()
    if args.stage == "check":   # counts only, no text
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        data = load_s()
        rows = []
        for e in data[:: max(1, len(data) // 60)]:
            for order in ("date", "relevance"):
                p, rec, trunc = build(e, tok, order, GEN)
                rows.append(dict(order=order, n=len(tok(p).input_ids), trunc=trunc, **rec))
        d = pd.DataFrame(rows)
        print(d.groupby("order").agg(n_med=("n", "median"), n_max=("n", "max"), trunc=("trunc", "mean"),
                                     recall_all=("recall_all", "mean")).round(2).to_string())
        return
    {"run": run, "judge": judge, "stats": stats}[args.stage](args)


if __name__ == "__main__":
    main()
