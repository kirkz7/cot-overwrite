"""MEME pilot (diagnosis only): does RAW-RAG-style retrieval surface the evidence that Cascade / Absence / Deletion
questions need? Design, split, thresholds: PREREG_MEME_PILOT.md (written before this ran).
Exploration episodes only (pl_001-015, sw_001-015); the other 70 are frozen held-out and are never loaded into `eps`.
Personal-life fillers come from LongMemEval: no conversation text is printed, aggregates only.
usage: python 论文/meme_pilot.py embed | rerank | stats      (run in this order; each stage is resumable)
"""
import glob
import json
import os
import sys

import numpy as np

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "meme_pilot")
DATA = glob.glob(r"D:\hf_cache\hub\datasets--meme-benchmark--MEME\snapshots\*\meme_filler32k.json")[0]
EMB = glob.glob(r"D:\hf_cache\hub\models--Qwen--Qwen3-Embedding-4B\snapshots\*")[0]
RER = glob.glob(r"D:\hf_cache\hub\models--Qwen--Qwen3-Reranker-4B\snapshots\*")[0]
EXPL = [f"{d}_{i:03d}" for d in ("pl", "sw") for i in range(1, 16)]
WIN, OVERLAP, DENSE_K, FINAL_K, BASE_K = 256, 32, 128, 15, 9
Q_TASK = "Given a web search query, retrieve relevant passages that answer the query"   # model-card default
R_INSTR = "Given a question and a passage, determine if the passage contains information relevant to answering the question."
RULES, CHANGE = "evidence_fact_introduction_part_2", "evidence_change+delete_event"


def load():
    eps = [e for e in json.load(open(DATA, encoding="utf-8")) if e["episode_id"] in EXPL]
    assert len(eps) == 30
    return eps


def serialize(ep):
    """history visible at the after-phase, one line per message; returns text and the char span of every message"""
    last = ep["after_questions"]["position_after_session"]
    text, spans = "", []          # spans: (session_idx, turn_idx, role, start, end)
    for si, s in enumerate(ep["sessions"]):
        if si > last:
            break
        text += f"[Session {si + 1} | {s['timestamp']}]\n"
        for ti, t in enumerate(s["conversation"]):
            line = f"{t['role']}: {t['content']}\n"
            spans.append((si, ti, t["role"], len(text), len(text) + len(line)))
            text += line
        text += "\n"
    return text, spans


def windows(tok, text):
    off = tok(text, add_special_tokens=False, return_offsets_mapping=True)["offset_mapping"]
    out, step = [], WIN - OVERLAP
    for a in range(0, max(1, len(off) - OVERLAP), step):
        b = min(a + WIN, len(off))
        out.append((off[a][0], off[b - 1][1]))
        if b == len(off):
            break
    return out


def user_turn(ep, spans, session_id, pred):
    """char span of the first user message in the named evidence session that satisfies pred"""
    for si, ti, role, a, b in spans:
        s = ep["sessions"][si]
        if s["session_id"] == session_id and role == "user" and pred(s["conversation"][ti]["content"]):
            return [a, b]
    return None


def required(ep, spans, q):
    """evidence contract of PREREG section 4: {name: char span or None}"""
    ent, tt, tgt = ep["entities"], q["task_type"], q["entity"][0]
    if tt == "Del":
        v = ent[tgt]["before"] if tgt in ent else list(q["entity_values"].values())[0]
        return {"delete": user_turn(ep, spans, CHANGE, lambda c: v in c and ("remove" in c.lower() or "delete" in c.lower()))}
    rule = lambda x: user_turn(ep, spans, RULES, lambda c: c.startswith("If ") and ent[x]["after"] in c)
    dep = lambda x: user_turn(ep, spans, RULES, lambda c: (not c.startswith("If ")) and ent[x]["before"] in c)
    need = {"trigger": user_turn(ep, spans, CHANGE, lambda c: ep["root_change"]["after"] in c)}
    need["target"] = rule(tgt) if tt == "Cas" else dep(tgt)
    if q.get("hop") == 2:
        mid = ent[tgt]["cascade_source"]
        need["middle"] = rule(mid) if ent[mid]["after"] else dep(mid)
    return need


def ancestors(ep, q):
    # 1-hop: the direct source (= root); 2-hop: the middle entity, then the root. The middle entity's own
    # cascade_source is not always filled in `entities`, so the root is taken from ep["root"] (fixed before any retrieval ran).
    src = ep["entities"].get(q["entity"][0], {}).get("cascade_source")
    if q["task_type"] == "Del" or not src:
        return []
    return [src] if q.get("hop") != 2 or src == ep["root"] else [src, ep["root"]]


def hop_query(ep, a):
    w = "my" if ep["domain"] == "personal_life" else "our"
    n = a.replace("_", " ")
    return f"What is {w} current {n}? Has {w} {n} changed?"


def questions(ep):
    return [q for q in ep["after_questions"]["questions"] if q["task_type"] in ("Cas", "Abs", "Del")]


def embed():
    import torch
    import torch.nn.functional as F
    from transformers import AutoModel, AutoTokenizer
    os.makedirs(OUT, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(EMB, padding_side="left")
    model = AutoModel.from_pretrained(EMB, torch_dtype=torch.float16).cuda().eval()

    def enc(texts, bs=16):
        res = []
        for i in range(0, len(texts), bs):
            b = tok(texts[i:i + bs], padding=True, truncation=True, max_length=1024, return_tensors="pt").to("cuda")
            with torch.no_grad():
                h = model(**b).last_hidden_state[:, -1]          # left padding: last token
            res.append(F.normalize(h.float(), p=2, dim=1).cpu())
        return torch.cat(res)

    for ep in load():
        path = os.path.join(OUT, f"dense_{ep['episode_id']}.json")
        if os.path.exists(path):
            continue
        text, spans = serialize(ep)
        wins = windows(tok, text)
        W = enc([text[a:b] for a, b in wins])
        qs = sorted({q["question"] for q in questions(ep)} | {hop_query(ep, a) for q in questions(ep) for a in ancestors(ep, q)})
        Q = enc([f"Instruct: {Q_TASK}\nQuery:{q}" for q in qs])
        sim = (Q @ W.T).numpy()
        top = {q: np.argsort(-sim[i])[:DENSE_K].tolist() for i, q in enumerate(qs)}
        json.dump(dict(windows=wins, dense=top), open(path, "w"))
        print(ep["episode_id"], "windows", len(wins), "queries", len(qs), flush=True)


def rerank():
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    tok = AutoTokenizer.from_pretrained(RER, padding_side="left")
    model = AutoModelForCausalLM.from_pretrained(RER, torch_dtype=torch.float16).cuda().eval()
    yes, no = tok.convert_tokens_to_ids("yes"), tok.convert_tokens_to_ids("no")
    pre = tok.encode("<|im_start|>system\nJudge whether the Document meets the requirements based on the Query and the Instruct "
                     "provided. Note that the answer can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n", add_special_tokens=False)
    suf = tok.encode("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n", add_special_tokens=False)

    def score(pairs, bs=16):
        res = []
        for i in range(0, len(pairs), bs):
            ids = tok([f"<Instruct>: {R_INSTR}\n<Query>: {q}\n<Document>: {d}" for q, d in pairs[i:i + bs]], padding=False,
                      truncation=True, max_length=1024 - len(pre) - len(suf), return_attention_mask=False)["input_ids"]
            b = tok.pad({"input_ids": [pre + x + suf for x in ids]}, padding=True, return_tensors="pt").to("cuda")
            with torch.no_grad():
                lg = model(**b).logits[:, -1, :]
            res += torch.log_softmax(torch.stack([lg[:, no], lg[:, yes]], 1).float(), 1)[:, 1].exp().tolist()
        return res

    for ep in load():
        path = os.path.join(OUT, f"rerank_{ep['episode_id']}.json")
        if os.path.exists(path):
            continue
        d = json.load(open(os.path.join(OUT, f"dense_{ep['episode_id']}.json")))
        text, _ = serialize(ep)
        out = {}
        for q, cand in d["dense"].items():
            s = score([(q, text[d["windows"][c][0]:d["windows"][c][1]]) for c in cand])
            out[q] = [cand[j] for j in np.argsort(-np.array(s), kind="stable")]     # dense candidates in rerank order
        json.dump(out, open(path, "w"))
        print(ep["episode_id"], "queries", len(out), flush=True)


def stats():
    rows = []
    for ep in load():
        text, spans = serialize(ep)
        d = json.load(open(os.path.join(OUT, f"dense_{ep['episode_id']}.json")))
        r = json.load(open(os.path.join(OUT, f"rerank_{ep['episode_id']}.json")))
        wins = d["windows"]
        hit = lambda sel, sp: any(wins[w][0] < sp[1] and sp[0] < wins[w][1] for w in sel)
        for q in questions(ep):
            need = required(ep, spans, q)
            row = dict(ep=ep["episode_id"], entity=q["entity"][0], task=q["task_type"], hop=q.get("hop", 0), located=all(v is not None for v in need.values()))
            if row["located"]:
                order = r[q["question"]]
                base = order[:FINAL_K]
                sel = order[:BASE_K]
                anc = ancestors(ep, q)
                for a in anc:                                    # equal budget: 9 + 6 split over the upstream queries
                    add = [w for w in r[hop_query(ep, a)] if w not in sel][:(FINAL_K - BASE_K) // len(anc)]
                    sel = sel + add
                for name, sp in need.items():
                    row[f"base_{name}"] = hit(base, sp)
                    row[f"hop_{name}"] = hit(sel, sp)
                    row[f"dense_{name}"] = hit(d["dense"][q["question"]], sp)      # anywhere in the 128 candidates
                for k in ("base", "hop", "dense"):
                    row[f"{k}_full"] = all(row[f"{k}_{n}"] for n in need)
                row["n_sel_hop"] = len(sel)
            rows.append(row)
    json.dump(rows, open(os.path.join(OUT, "per_question.json"), "w"), indent=1)
    import pandas as pd
    df = pd.DataFrame(rows)
    print("questions", len(df), "| evidence not located (dropped):", df[~df.located].groupby("task").size().to_dict())
    df = df[df.located]
    rng, eps = np.random.default_rng(0), sorted(df.ep.unique())

    def ci(g, col, col2=None):
        """episode bootstrap of mean(col) (or mean(col) - mean(col2)) in percent"""
        y = g[col].astype(float) - (g[col2].astype(float) if col2 else 0)
        s = y.groupby(g.ep).agg(["sum", "count"]).reindex(eps, fill_value=0).values      # per-episode sum and n
        idx = rng.integers(0, len(eps), (10000, len(eps)))
        vals = s[idx, 0].sum(1) / np.maximum(s[idx, 1].sum(1), 1) * 100
        return f"[{np.percentile(vals, 2.5):.0f}, {np.percentile(vals, 97.5):.0f}]"

    for task in ("Cas", "Abs", "Del"):
        g = df[df.task == task]
        print(f"\n== {task} n={len(g)} (1-hop {int((g.hop == 1).sum())}, 2-hop {int((g.hop == 2).sum())})")
        comps = ["delete"] if task == "Del" else ["target", "trigger", "middle", "full"]
        for c in comps:
            x = g.dropna(subset=[f"base_{c}"]) if f"base_{c}" in g else g.iloc[:0]
            if not len(x):
                continue
            x = x.astype({f"base_{c}": float, f"hop_{c}": float, f"dense_{c}": float})
            line = (f"  {c:8s} n={len(x):2d} | in dense-128 {x[f'dense_{c}'].mean() * 100:5.1f} | top-15 {x[f'base_{c}'].mean() * 100:5.1f} "
                    f"{ci(x, f'base_{c}')}")
            if task != "Del":
                line += (f" | 9+6 upstream {x[f'hop_{c}'].mean() * 100:5.1f} {ci(x, f'hop_{c}')} | diff "
                         f"{(x[f'hop_{c}'].mean() - x[f'base_{c}'].mean()) * 100:+5.1f} {ci(x, f'hop_{c}', f'base_{c}')}")
            print(line)
        if task != "Del":
            for h in (1, 2):
                x = g[g.hop == h]
                if len(x):
                    print(f"  full path, {h}-hop n={len(x)}: top-15 {x.base_full.mean() * 100:.1f} | 9+6 upstream {x.hop_full.mean() * 100:.1f}")
            x = g.astype({"base_target": float, "base_trigger": float})
            print(f"  P2: trigger minus target coverage (top-15) {(x.base_trigger.mean() - x.base_target.mean()) * 100:+.1f} {ci(x, 'base_trigger', 'base_target')}")


if __name__ == "__main__":
    {"embed": embed, "rerank": rerank, "stats": stats}[sys.argv[1]]()
