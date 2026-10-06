"""Held-out evaluations for the order-vs-time fine-tune (PREREG_TRAINING.md). Frozen before any trained model is
evaluated; every model (base and LoRA-tuned, "Base@runs/x/final") is run with exactly this code. Resumable.

  mab        MemoryAgentBench conflict resolution (FactConsolidation single-hop / multi-hop, 6k): numbered facts where a
             larger serial number is newer; facts shown in original, reversed or shuffled order (100 + 100 questions)
  tot        Test of Time (ToT-Semantic): same questions with facts sorted by start time vs shuffled
             (question types event_at_time_t, first_last, before_after; 70 questions each)
  tempreason TempReason L2 (time-interval facts, "which X in <time>?"): facts sorted / reversed / shuffled (300 questions)
  babilong   BABILong qa1-qa3 at 2k (state tracking without dates; regression check of the default reading; 150 each)
  gsm8k      GSM8K test (general reasoning regression; 250 problems, greedy step-by-step)
  cot        synthetic CoT state tracking with explicit order cues (Exp 18 conditions; k = 2, 4, 8; 100 each).
             Note: its rev_header condition uses a "newest first" header, a cue type that IS trained (undated samples).
usage: python run_ext_eval.py --models Qwen3-4B,Qwen3-4B@runs/q4-dec-s0/final --benches mab,tot,tempreason,babilong,cot,gsm8k
       python run_ext_eval.py check
"""
import argparse
import ast
import collections
import json
import random
import re
import zlib

import pandas as pd
import torch
from huggingface_hub import hf_hub_download
from tqdm import tqdm

from app_common import JsonlAppender, chat_prompt, free_gpu, load_reader, tag
from probe import PREFILL, greedy
from torch.nn.attention import SDPBackend, sdpa_kernel

BENCHES = ["mab", "tot", "tempreason", "babilong", "cot", "gsm8k"]
MAB_INSTR = ("Each fact below has a serial number. A fact with a larger serial number is newer, and it overrides any older "
             "fact it conflicts with. Answer the question using only these facts, not real-world knowledge.")


def norm(s):
    return " ".join(re.sub(r"[^0-9a-z\- ]", " ", str(s).casefold()).split())


def contains_any(resp, golds):
    r = f" {norm(resp)} "
    return any(f" {norm(g)} " in r for g in golds if norm(g))


def stop_ids(tok):
    vocab = tok.get_vocab()
    return {tok.eos_token_id} | {vocab[t] for t in ("<|im_end|>", "<|end|>", "<|eot_id|>", "<|endoftext|>") if t in vocab}


def gen(tok, model, user, max_new=32):
    ids = torch.tensor([tok(chat_prompt(tok, user), add_special_tokens=False).input_ids], device="cuda")
    return tok.decode(greedy(model, ids, max_new, stop_ids(tok)), skip_special_tokens=True).strip()


# ---------------------------------------------------------------- items (no model needed)
def mab_items():
    df = pd.read_parquet(hf_hub_download("ai-hyz/MemoryAgentBench", "data/Conflict_Resolution-00000-of-00001.parquet",
                                         repo_type="dataset"))
    out = []
    for _, r in df.iterrows():
        src = r["metadata"]["source"]
        if src not in ("factconsolidation_sh_6k", "factconsolidation_mh_6k"):
            continue
        facts = [(int(m.group(1)), m.group(2)) for m in re.finditer(r"^(\d+)\. (.*)$", r["context"], re.M)]
        for qi, (q, a) in enumerate(zip(r["questions"], r["answers"])):
            out.append(dict(src=src, qi=qi, q=q, golds=list(a), facts=facts))
    return out


def mab_context(facts, order, seed):
    fs = facts[:] if order == "original" else facts[::-1] if order == "reversed" else random.Random(seed).sample(facts, len(facts))
    return MAB_INSTR + "\n\nHere is a list of facts:\n" + "\n".join(f"{i}. {t}" for i, t in fs)


def tot_items():
    df = pd.read_parquet(hf_hub_download("baharef/ToT", "tot_semantic/test/0000.parquet", repo_type="dataset",
                                         revision="refs/convert/parquet"))
    df = df[df.question_type.isin(["event_at_time_t", "first_last", "before_after"]) &
            df.sorting_type.isin(["start_time_and_target", "shuffle"])]
    out = []
    for _, r in df.iterrows():
        facts = re.split(r"\s*Answer the following question", r["prompt"])[0].strip()   # drop ToT's JSON-output instruction
        assert "JSON" not in facts and "Answer the following" not in facts
        user = (facts + "\n\nAnswer the following question based on the temporal facts above. Answer with the entity or "
                "value only.\nQuestion: " + r["question"])
        out.append(dict(qtype=r["question_type"], order="sorted" if r["sorting_type"] != "shuffle" else "shuffle",
                        question=r["question"], user=user, gold=str(r["label"])))
    return out


MONTHS = {m: i for i, m in enumerate(["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"], 1)}


def fact_start(f):
    m = re.search(r"from (\w{3})\w*,? (\d{4})", f)
    return (int(m.group(2)), MONTHS.get(m.group(1)[:3], 0)) if m else (9999, 0)


def tempreason_items(n=300):
    lines = open(hf_hub_download("tonytan48/TempReason", "test_l2.json", repo_type="dataset"), encoding="utf-8").read().splitlines()
    rows = [json.loads(l) for l in random.Random(7).sample(lines, n)]
    out = []
    for r in rows:
        facts = [f.strip() for f in r["fact_context"].split("\n") if f.strip()]
        lit = lambda x: ast.literal_eval(x) if isinstance(x, str) else x      # stored either as text or as objects
        out.append(dict(id=r["id"], question=r["question"], facts=sorted(facts, key=fact_start),
                        golds=lit(r["text_answers"])["text"], negs=list(lit(r["neg_answers"]) or [])))
    return out


def babilong_items(n=150):
    out = []
    for qa in ("qa1", "qa2", "qa3"):
        df = pd.read_parquet(hf_hub_download("RMT-team/babilong-1k-samples", f"2k/{qa}/0000.parquet", repo_type="dataset",
                                             revision="refs/convert/parquet"))
        for i in random.Random(8).sample(range(len(df)), n):
            r = df.iloc[i]
            out.append(dict(task=qa, i=int(i), user=f"{r['input']}\n\nQuestion: {r['question'].strip()}\nAnswer with a single word.",
                            gold=r["target"]))
    return out


def gsm8k_items(n=250):
    df = pd.read_parquet(hf_hub_download("openai/gsm8k", "main/test/0000.parquet", repo_type="dataset", revision="refs/convert/parquet"))
    return [dict(i=i, user=df.question[i] + "\nSolve the problem step by step, then give the final answer on a last line "
                 "in the form 'The answer is N'.", gold=float(df.answer[i].split("####")[-1].replace(",", "").strip()))
            for i in range(n)]


# ---------------------------------------------------------------- runners
@torch.no_grad()
def run_mab(tok, model, w):
    items = mab_items()
    jobs = [(src, order) for src in sorted({it["src"] for it in items}) for order in ("original", "reversed", "shuffled")]
    for src, order in tqdm(jobs, desc="mab"):
        group = [it for it in items if it["src"] == src]
        todo = [it for it in group if (src, order, it["qi"]) not in w.done]
        if not todo:
            continue
        ctx = mab_context(group[0]["facts"], order, seed=zlib.crc32(src.encode()))   # same shuffle for every model / run
        marker = "\n\nQuestion: "
        full = lambda q: chat_prompt(tok, f"{ctx}{marker}{q}\nAnswer with the answer only.")
        if getattr(model, "is_vllm", False):   # vLLM: full prompt each time (the server reuses the shared prefix itself)
            for it in todo:
                out = greedy(model, torch.tensor([tok(full(it["q"]), add_special_tokens=False).input_ids]), 32, stop_ids(tok))
                resp = tok.decode(out, skip_special_tokens=True).strip().split("\n")[0]
                w.write(dict(src=src, order=order, qi=it["qi"], response=resp, correct=contains_any(resp, it["golds"])))
            continue
        prefix_text = full("X").split(marker)[0] + marker
        p_ids = tok(prefix_text, add_special_tokens=False).input_ids
        with sdpa_kernel(PREFILL, set_priority=True):
            cache = model(torch.tensor([p_ids], device="cuda"), use_cache=True).past_key_values
        stops = stop_ids(tok)
        for it in todo:
            ids = tok(full(it["q"]), add_special_tokens=False).input_ids
            if ids[:len(p_ids)] != p_ids:                         # tokenisation boundary moved: full prefill instead
                out = greedy(model, torch.tensor([ids], device="cuda"), 32, stops)
            else:
                cache.crop(len(p_ids))
                with sdpa_kernel(SDPBackend.MATH):
                    o = model(torch.tensor([ids[len(p_ids):]], device="cuda"), past_key_values=cache, use_cache=True)
                    out, nxt = [], o.logits[0, -1].argmax().item()
                    for _ in range(32):
                        out.append(nxt)
                        if nxt in stops:
                            break
                        o = model(torch.tensor([[nxt]], device="cuda"), past_key_values=cache, use_cache=True)
                        nxt = o.logits[0, -1].argmax().item()
            resp = tok.decode(out, skip_special_tokens=True).strip().split("\n")[0]
            w.write(dict(src=src, order=order, qi=it["qi"], response=resp, correct=contains_any(resp, it["golds"])))
        del cache
        torch.cuda.empty_cache()


def tot_score(it, resp):
    """ToT answers here are entity ids (E76): the first entity id in the answer line must be the gold one."""
    first = resp.split("\n")[0]
    if re.fullmatch(r"E\d+", it["gold"].strip()):
        ents = re.findall(r"\bE\d+\b", first)
        return dict(correct=ents[:1] == [it["gold"].strip()])
    return dict(correct=contains_any(first, [it["gold"]]))


def run_simple(tok, model, w, items, key, max_new, score):
    limit = getattr(model.config, "max_position_embeddings", 10 ** 9) if getattr(model, "is_vllm", False) else 10 ** 9
    for it in tqdm([x for x in items if key(x) not in w.done]):
        if len(tok(chat_prompt(tok, it["user"]), add_special_tokens=False).input_ids) + max_new > limit:
            # vLLM cannot place tokens past the trained length (RoPE table; 4 ToT prompts reach 42k): recorded, not scored
            w.write(dict(**{k: v for k, v in it.items() if k not in ("user", "facts")}, response="", correct=None, skipped=True))
            continue
        resp = gen(tok, model, it["user"], max_new)
        w.write(dict(**{k: v for k, v in it.items() if k not in ("user", "facts")}, response=resp[-400:], **score(it, resp)))


def run_tempreason(tok, model, w):
    items = tempreason_items()
    jobs = [(it, o) for it in items for o in ("sorted", "reversed", "shuffled")]
    for it, o in tqdm([j for j in jobs if (j[0]["id"], j[1]) not in w.done], desc="tempreason"):
        fs = it["facts"] if o == "sorted" else it["facts"][::-1] if o == "reversed" else random.Random(it["id"]).sample(it["facts"], len(it["facts"]))
        resp = gen(tok, model, "Here are some facts:\n" + "\n".join(fs) + f"\n\nQuestion: {it['question']}\nAnswer with the answer only.")
        first = resp.split("\n")[0]
        ok = contains_any(first, it["golds"])
        w.write(dict(id=it["id"], order=o, n_facts=len(fs), response=first, correct=ok,
                     wrong_time=(not ok) and contains_any(first, it["negs"])))


def run_cot(tok, model, w):
    assert not getattr(model, "is_vllm", False), "cot scores candidate log-probs on the HF model: run --benches cot without COT_ENGINE=vllm"
    from probe import Scorer
    from tasks import make_example
    from tasks_cue import CONDS, build_cue_prompt
    scorer = Scorer(tok, model)
    for k in (2, 4, 8):
        for i in tqdm(range(100), desc=f"cot k={k}"):
            ex = make_example(k, seed=k * 100_000 + i)
            hist = ex.history(ex.target)
            for c in CONDS:
                if (k, i, c) in w.done:
                    continue
                p, seen = build_cue_prompt(ex, ex.target, c, tok, "sym", False)
                s = scorer.score(p)
                pred = max(s, key=s.get)
                w.write(dict(k=k, i=i, cond=c, correct=pred == hist[-1], pick_last_presented=pred == seen[-1],
                             gold_is_last_presented=hist[-1] == seen[-1]))


def run_gsm8k(tok, model, w, four_bit):
    items = [it for it in gsm8k_items() if it["i"] not in w.done]
    if not items:
        return
    def score(it, text):
        m = re.findall(r"answer is\s*\$?\s*(-?[\d,]*\.?\d+)", text.replace("**", ""))
        nums = m or re.findall(r"-?\d[\d,]*\.?\d*", text)
        try:
            pred = float(nums[-1].replace(",", "")) if nums else None
        except ValueError:
            pred = None
        return dict(pred=pred, correct=pred is not None and abs(pred - it["gold"]) < 1e-6)
    # CUDA-graph decoding only for architectures verified to be capturable (Phi-4-mini's LongRoPE length switch is not)
    if four_bit or getattr(model, "is_vllm", False) or model.config.model_type not in ("qwen3", "qwen2"):
        for it in tqdm(items, desc="gsm8k"):
            text = gen(tok, model, it["user"], 400)
            w.write(dict(i=it["i"], response=text[-300:], **score(it, text)))
        return
    from fastgen import GraphGen, auto_batch
    enc = [tok(chat_prompt(tok, it["user"]), add_special_tokens=False).input_ids for it in items]
    L = max(map(len, enc)) + 400 + 8
    gg = GraphGen(model, auto_batch(model, L, cap=8), L)
    stops = stop_ids(tok)
    for b in tqdm(range(0, len(items), gg.B), desc="gsm8k"):
        for it, (row, _) in zip(items[b:b + gg.B], gg.generate(enc[b:b + gg.B], 400, stops, tok.pad_token_id, sample=False)):
            text = tok.decode(row, skip_special_tokens=True)
            w.write(dict(i=it["i"], response=text[-300:], **score(it, text)))
    model.set_attn_implementation("sdpa")


def summarize(bench, rows):
    df = pd.DataFrame(rows)
    if "skipped" in df:                                       # vLLM: prompts past the trained length, not scored
        df = df[df.skipped != True].astype({"correct": bool})
    if df.empty:
        return ""
    by = {"mab": ["src", "order"], "tot": ["qtype", "order"], "tempreason": ["order"], "babilong": ["task"],
          "cot": ["cond", "k"], "gsm8k": []}[bench]
    return (df.groupby(by).correct.mean() * 100).round(1).to_string() if by else f"accuracy {df.correct.mean() * 100:.1f}"


KEYS = {"mab": lambda r: (r["src"], r["order"], r["qi"]), "tot": lambda r: (r["question"], r["order"]),
        "tempreason": lambda r: (r["id"], r["order"]), "babilong": lambda r: (r["task"], r["i"]),
        "cot": lambda r: (r["k"], r["i"], r["cond"]), "gsm8k": lambda r: r["i"]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", nargs="?", default="run", choices=["run", "check"])
    ap.add_argument("--models", default="Qwen3-4B")
    ap.add_argument("--benches", default=",".join(BENCHES))
    args = ap.parse_args()
    if args.stage == "check":
        from transformers import AutoTokenizer
        tok = AutoTokenizer.from_pretrained("Qwen/Qwen3-4B")
        m = mab_items()
        print("mab questions", len(m), "| context tokens", len(tok(mab_context(m[0]["facts"], "reversed", 1)).input_ids),
              "| facts", len(m[0]["facts"]), "| reversed starts with", mab_context(m[0]["facts"], "reversed", 1).split("\n")[3][:6])
        t = tot_items()
        print("tot prompts", len(t), dict(collections.Counter((x["qtype"], x["order"]) for x in t)),
              "| median tokens", sorted(len(tok(x["user"]).input_ids) for x in t[:200])[100])
        tr = tempreason_items()
        print("tempreason questions", len(tr), "| facts per question (median)", sorted(len(x["facts"]) for x in tr)[150],
              "| facts without a parseable start:", sum(fact_start(f)[0] == 9999 for x in tr for f in x["facts"]),
              "of", sum(len(x["facts"]) for x in tr), "| gold in facts:", sum(any(g in " ".join(x["facts"]) for g in x["golds"]) for x in tr))
        print("tot golds that are entity ids:", sum(bool(re.fullmatch(r"E\d+", x["gold"].strip())) for x in t), "of", len(t),
              "| tot_score('E76 was...'):", tot_score(dict(gold="E76"), "The entity is E76."), tot_score(dict(gold="E76"), "E44"))
        b = babilong_items()
        print("babilong", len(b), dict(collections.Counter(x["task"] for x in b)))
        g = gsm8k_items()
        print("gsm8k", len(g), "| gold example", g[0]["gold"])
        return
    for name in args.models.split(","):
        base = name.split("@")[0]
        tok = model = None
        for bench in args.benches.split(","):
            w = JsonlAppender(f"results/ext_{bench}_{tag(name)}.jsonl", key=KEYS[bench])
            expected = {"mab": 600, "tot": 420, "tempreason": 900, "babilong": 450, "cot": 2100, "gsm8k": 250}[bench]
            if len(w.done) >= expected:
                print(name, bench, "already complete", flush=True)
                w.close()
                continue
            if model is None:
                tok, model = load_reader(name)
            if bench == "mab":
                run_mab(tok, model, w)
            elif bench == "tot":
                run_simple(tok, model, w, tot_items(), KEYS["tot"], 24, tot_score)
            elif bench == "tempreason":
                run_tempreason(tok, model, w)
            elif bench == "babilong":
                run_simple(tok, model, w, babilong_items(), KEYS["babilong"], 8,
                           lambda it, resp: dict(correct=contains_any(resp.split("\n")[0], [it["gold"]])))
            elif bench == "cot":
                run_cot(tok, model, w)
            else:
                from app_common import MODELS
                run_gsm8k(tok, model, w, MODELS[base][1])
            w.close()
            print(f"== {name} | {bench}\n{summarize(bench, w.rows)}", flush=True)
        if model is not None:
            del tok, model
            free_gpu()


if __name__ == "__main__":
    main()
