"""LoRA SFT for the order-vs-time fix (PREREG_TRAINING.md). Resumable after crashes / power cuts.

Loss only on the answer tokens (+ end-of-turn token); logits are computed only for those positions.
Checkpoints (adapter + optimizer + scheduler + data position + RNG) every --save_every optimizer steps,
written atomically; rerunning the same command continues from the last checkpoint.
Validation: greedy exact match on a fixed subset of the synthetic validation set, by (dated, order, qtype).
The synthetic validation set is the only data used for any model-selection decision.
usage: python train_lora.py --model Qwen3-4B --data data_train/decoupled_train.jsonl --val data_train/decoupled_val.jsonl --out runs/qwen3-4b-dec
"""
import argparse
import collections
import json
import math
import os
import random
import re
import shutil
import time

import torch
import torch.nn.functional as F
import transformers.integrations.sdpa_attention as hf_sdpa
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from app_common import MODELS, chat_prompt, load_jsonl
from probe import greedy

# Repeat K/V instead of SDPA's enable_gqa: the memory-efficient kernel (O(L) memory, has a backward pass) does
# not accept enable_gqa on this torch build, and the fallback math kernel would need O(L^2) memory per layer.
hf_sdpa.use_gqa_in_sdpa = lambda *a, **k: False
TRAIN_SDPA = [SDPBackend.EFFICIENT_ATTENTION, SDPBackend.CUDNN_ATTENTION, SDPBackend.MATH]
# every linear layer except the LM head: architectures name their projections differently (Phi-4-mini fuses them into
# qkv_proj / gate_up_proj, so an explicit q_proj/.../up_proj list would silently adapt only o_proj and down_proj there)
TARGETS = "all-linear"


def end_of_turn(tok):
    tpl = tok.chat_template or ""
    for t in ("<|im_end|>", "<|end|>", "<|eot_id|>"):
        if t in tok.get_vocab() and t in tpl:
            return t
    return tok.eos_token


def encode(tok, row, eot):
    prefix = tok(chat_prompt(tok, row["prompt"], row.get("prefix", "")), add_special_tokens=False).input_ids   # optional assistant prefix (CoT data)
    ans = tok(row["answer"] + eot, add_special_tokens=False).input_ids
    return prefix, ans


def norm(s):
    """Lenient exact match: case, quotes, currency signs and trailing punctuation do not matter ("$460" == "460")."""
    return " ".join(re.sub(r"[^0-9a-z\-. ]", " ", s.casefold()).strip(" .").split())


@torch.no_grad()
def validate(model, tok, val, eot_ids, max_new=24):
    if not val:
        return {}
    model.eval()
    hits = collections.defaultdict(list)
    for r in val:
        ids = torch.tensor([tok(chat_prompt(tok, r["prompt"], r.get("prefix", "")), add_special_tokens=False).input_ids], device="cuda")
        out = tok.decode(greedy(model, ids, max_new, eot_ids), skip_special_tokens=True).strip().split("\n")[0]
        ok = norm(out) == norm(r["answer"])
        hits["all"].append(ok)
        hits[f"dated={r['dated']}|order={r['order']}"].append(ok)
        hits[f"qtype={r['qtype']}"].append(ok)
    model.train()
    return {k: round(100 * sum(v) / len(v), 1) for k, v in sorted(hits.items())} | {"n": len(val)}


def save_ckpt(model, opt, sched, state, path):
    tmp = path + ".tmp"
    shutil.rmtree(tmp, ignore_errors=True)
    model.save_pretrained(tmp)
    torch.save(dict(opt=opt.state_dict(), sched=sched.state_dict(), state=state, py_rng=random.getstate(),
                    torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state()), os.path.join(tmp, "trainer_state.pt"))
    old = path + ".old"
    shutil.rmtree(old, ignore_errors=True)
    if os.path.exists(path):
        os.replace(path, old)
    os.replace(tmp, path)
    shutil.rmtree(old, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--rank", type=int, default=16)
    ap.add_argument("--alpha", type=int, default=32)
    ap.add_argument("--dropout", type=float, default=0.05)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--epochs", type=float, default=1.0)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--max_len", type=int, default=6144)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--eval_every", type=int, default=200)
    ap.add_argument("--val_n", type=int, default=150)
    ap.add_argument("--dev", default=None, help="held-out-format dev set: the only set used for hyperparameter decisions")
    ap.add_argument("--dev_n", type=int, default=300)
    ap.add_argument("--limit", type=int, default=None, help="use only the first N training rows (smoke tests)")
    ap.add_argument("--seed", type=int, default=0, help="data order and LoRA initialisation")
    ap.add_argument("--probe", action="store_true", help="memory probe: 3 steps on the longest samples, then exit")
    args = ap.parse_args()
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training, set_peft_model_state_dict
    from peft.utils import load_peft_weights

    os.makedirs(args.out, exist_ok=True)
    json.dump(vars(args), open(os.path.join(args.out, "args.json"), "w"), indent=1)
    hf, four_bit = MODELS.get(args.model, (args.model, False))
    tok = AutoTokenizer.from_pretrained(hf)
    kw = dict(dtype=torch.bfloat16, device_map="cuda", attn_implementation="sdpa")
    if four_bit:
        kw["quantization_config"] = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16)
    model = AutoModelForCausalLM.from_pretrained(hf, **kw)
    model.config.use_cache = False
    if four_bit:
        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=True,
                                                gradient_checkpointing_kwargs={"use_reentrant": False})
    else:
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        model.enable_input_require_grads()
    torch.manual_seed(args.seed)                                 # LoRA initialisation depends on the seed
    model = get_peft_model(model, LoraConfig(r=args.rank, lora_alpha=args.alpha, lora_dropout=args.dropout,
                                             target_modules=TARGETS, task_type="CAUSAL_LM"))
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    eot = end_of_turn(tok)
    eot_ids = {tok.convert_tokens_to_ids(eot), tok.eos_token_id}

    rows = load_jsonl(args.data)[:args.limit]
    enc = [encode(tok, r, eot) for r in rows]
    keep = [i for i, (p, a) in enumerate(enc) if len(p) + len(a) <= args.max_len]
    rng = random.Random(args.seed)
    n_samples = int(len(keep) * args.epochs)
    order = [keep[i % len(keep)] for i in range(n_samples)]
    rng.shuffle(order)
    total_steps = math.ceil(n_samples / args.accum)
    val = load_jsonl(args.val)
    val = random.Random(123).sample(val, min(args.val_n, len(val)))
    dev = load_jsonl(args.dev) if args.dev else []
    dev = random.Random(456).sample(dev, min(args.dev_n, len(dev)))
    if args.probe:                                               # peak memory on the longest kept samples
        longest = sorted(keep, key=lambda i: -(len(enc[i][0]) + len(enc[i][1])))[:3]
        model.train()
        t = time.time()
        for i in longest:
            p, a = enc[i]
            ids = torch.tensor([p + a], device="cuda")
            with sdpa_kernel(TRAIN_SDPA, set_priority=True):
                logits = model(input_ids=ids, logits_to_keep=len(a) + 1).logits[0, :-1].float()
            F.cross_entropy(logits, ids[0, -len(a):]).backward()
        torch.cuda.synchronize()
        n_tok = sum(len(enc[i][0]) + len(enc[i][1]) for i in longest)
        free, total = torch.cuda.mem_get_info()
        print(json.dumps(dict(probe=args.model, longest_tokens=[len(enc[i][0]) + len(enc[i][1]) for i in longest],
                              peak_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2),
                              reserved_gb=round(torch.cuda.memory_reserved() / 2**30, 2),
                              free_gb_after=round(free / 2**30, 2), tok_s=round(n_tok / (time.time() - t)))), flush=True)
        return

    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.0, betas=(0.9, 0.999))
    warm = max(1, int(0.03 * total_steps))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total_steps - warm))))
    state = dict(step=0, seen=0, tokens=0)
    ckpt = os.path.join(args.out, "ckpt")
    if os.path.exists(os.path.join(ckpt, "trainer_state.pt")):
        set_peft_model_state_dict(model, load_peft_weights(ckpt))
        ts = torch.load(os.path.join(ckpt, "trainer_state.pt"), weights_only=False)
        opt.load_state_dict(ts["opt"]); sched.load_state_dict(ts["sched"]); state = ts["state"]
        random.setstate(ts["py_rng"]); torch.set_rng_state(ts["torch_rng"]); torch.cuda.set_rng_state(ts["cuda_rng"])
        print(f"resumed at step {state['step']} ({state['seen']} samples)", flush=True)
    log = open(os.path.join(args.out, "log.jsonl"), "a", encoding="utf-8")
    print(f"model {args.model} | trainable {n_train / 1e6:.1f}M | samples {n_samples} (dropped {len(enc) - len(keep)} > {args.max_len} tok)"
          f" | steps {total_steps}", flush=True)
    if state["step"] == 0:
        v, dv = validate(model, tok, val, eot_ids), validate(model, tok, dev, eot_ids)
        log.write(json.dumps(dict(step=0, val=v, dev=dv)) + "\n"); log.flush()
        print("step 0 val", v, "| dev", dv, flush=True)

    model.train()
    t0, run_loss, run_n = time.time(), 0.0, 0
    while state["seen"] < n_samples:
        batch = order[state["seen"]:state["seen"] + args.accum]
        for i in batch:
            p, a = enc[i]
            ids = torch.tensor([p + a], device="cuda")
            with sdpa_kernel(TRAIN_SDPA, set_priority=True):
                logits = model(input_ids=ids, logits_to_keep=len(a) + 1).logits[0, :-1].float()
            loss = F.cross_entropy(logits, ids[0, -len(a):])
            (loss / len(batch)).backward()
            run_loss += loss.item(); run_n += 1
            state["tokens"] += len(p) + len(a)
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step(); sched.step(); opt.zero_grad(set_to_none=True)
        state["seen"] += len(batch); state["step"] += 1
        if state["step"] % 10 == 0 or state["seen"] >= n_samples:
            el = time.time() - t0
            rec = dict(step=state["step"], loss=round(run_loss / max(run_n, 1), 4), lr=sched.get_last_lr()[0],
                       tok_s=round(state["tokens"] / max(el, 1e-6)), mem_gb=round(torch.cuda.max_memory_allocated() / 2**30, 2))
            log.write(json.dumps(rec) + "\n"); log.flush()
            print(rec, flush=True)
            run_loss, run_n = 0.0, 0
        if state["step"] % args.save_every == 0 or state["seen"] >= n_samples:
            save_ckpt(model, opt, sched, state, ckpt)
        if state["step"] % args.eval_every == 0 or state["seen"] >= n_samples:
            v, dv = validate(model, tok, val, eot_ids), validate(model, tok, dev, eot_ids)
            log.write(json.dumps(dict(step=state["step"], val=v, dev=dv)) + "\n"); log.flush()
            print("step", state["step"], "val", v, "| dev", dv, flush=True)
    final = os.path.join(args.out, "final")
    if not os.path.exists(final):
        tmp = final + ".tmp"
        shutil.rmtree(tmp, ignore_errors=True)
        model.save_pretrained(tmp)
        os.replace(tmp, final)
    print("done; adapter at", final, flush=True)


if __name__ == "__main__":
    main()
