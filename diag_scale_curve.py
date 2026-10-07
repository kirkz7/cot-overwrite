"""Review 10-07 (diagnosis, no scoring change): scale curve over every test set from the cloud result files (Qwen3
4B / 8B / 14B / 32B on vLLM bf16, Gemma-3-12B, Qwen3-14B 4-bit on HF), and the 10-06 pre-registered scale rule
(EXPLORE_PLAN "10-06 复盘新增" A: 32B effect minus 4B effect on the same items). Aggregates only, no conversation or
response text is printed. Runs without torch (reads jsonl only), e.g. on a CPU-only checkout of the cloud-l20 branch.
usage: python diag_scale_curve.py [results dir]
"""
import json
import os
import sys

import numpy as np
import pandas as pd

R = sys.argv[1] if len(sys.argv) > 1 else "results"
B = 10000
MODELS = ["Qwen3-4B~vllm", "Qwen3-8B-bf16~vllm", "Qwen3-14B-bf16~vllm", "Qwen3-32B~vllm", "Gemma-3-12B~vllm", "Qwen3-14B"]


def load(name):
    p = os.path.join(R, name)
    if not os.path.exists(p) or not os.path.getsize(p):
        return pd.DataFrame()
    rows = []
    for line in open(p, encoding="utf-8"):
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            pass
    return pd.DataFrame(rows)


def boot(d, groups=None, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    else:
        g = pd.Series(list(groups)).astype("category").cat.codes.values
        k = g.max() + 1
        s, c = np.bincount(g, weights=d, minlength=k), np.bincount(g, minlength=k)
        idx = rng.integers(0, k, (B, k))
        m = s[idx].sum(1) / c[idx].sum(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


def f(t):
    return f"{t[0]:+5.1f} [{t[1]:+5.1f}, {t[2]:+5.1f}]"


def tag(m):
    return m.replace("~vllm", "").replace("-bf16", "")


print("#### MemConflict (judge v1; Gemma: v2 where v1 missing)")
for m in MODELS:
    raw = load(f"extmem_{m}.jsonl")
    if raw.empty:
        continue
    raw = raw[raw.task == "memconf"]
    j = load(f"extmem_{m}_judged.jsonl")
    src = "v1"
    if j.empty or "memconf" not in set(j.task):
        j, src = load(f"extmem_{m}_judged2.jsonl"), "v2"
    if j.empty:
        print(f"  {tag(m)}: not judged")
        continue
    j = j[j.task == "memconf"][["id", "cond", "label"]]
    df = raw.merge(j, on=["id", "cond"])
    p = df.pivot_table(index="id", columns="cond", values="label", aggfunc="first").dropna()
    if len(p) < 30:
        print(f"  {tag(m)}: only {len(p)} items judged in all orders ({src})")
        continue
    ok = (p == "A").astype(float)
    st = (p == "B").astype(float)
    cl = p.index.str[:8]
    print(f"  {tag(m):12s} {src} n={len(p)} acc {ok.chrono.mean()*100:5.1f}/{ok.rev.mean()*100:5.1f}/{ok.retr.mean()*100:5.1f} "
          f"stale {st.chrono.mean()*100:4.1f}/{st.rev.mean()*100:4.1f}/{st.retr.mean()*100:4.1f} | rev {f(boot(ok.rev-ok.chrono))} "
          f"cl {f(boot(ok.rev-ok.chrono, cl))} | retr {f(boot(ok.retr-ok.chrono))} cl {f(boot(ok.retr-ok.chrono, cl))}")

print("\n#### LongMemEval knowledge update (seen), judged")
for m in MODELS:
    j = load(f"app_memory_{m}_judged.jsonl")
    if j.empty:
        continue
    j = j[~j.skipped.astype(bool)] if "skipped" in j else j
    ok = j.assign(c=(j.label == "A").astype(float)).pivot_table(index="qid", columns="cond", values="c", aggfunc="first")
    st = j.assign(c=(j.label == "B").astype(float)).pivot_table(index="qid", columns="cond", values="c", aggfunc="first")
    def pair(a, b):
        x = ok[[a, b]].dropna()
        return f"{x[b].mean()*100:5.1f}->{x[a].mean()*100:5.1f} {f(boot(x[a]-x[b]))}"
    print(f"  {tag(m):12s} S dated {pair('S_rev_dated','S_chrono_dated')} | +header {ok.S_rev_dated_header.mean()*100:5.1f} | "
          f"nodate {pair('S_rev_nodate','S_chrono_nodate')} | T dated {pair('T_oldlast_dated','T_newlast_dated')} | "
          f"stale S {st.S_chrono_dated.mean()*100:4.1f}->{st.S_rev_dated.mean()*100:4.1f}")

print("\n#### ConvoMem-long (judged) / PersonaMem (letter)")
for m in MODELS:
    raw = load(f"longconv_{m}.jsonl")
    if raw.empty:
        continue
    j = load(f"longconv_{m}_judged.jsonl")
    out = f"  {tag(m):12s}"
    if not j.empty:
        ok = j.assign(c=(j.label == "A").astype(float)).pivot_table(index="id", columns="cond", values="c", aggfunc="first").dropna()
        st = j.assign(c=(j.label == "B").astype(float)).pivot_table(index="id", columns="cond", values="c", aggfunc="first").dropna()
        out += f" CML n={len(ok)} {ok.chrono.mean()*100:5.1f}/{ok.rev.mean()*100:5.1f} {f(boot(ok.rev-ok.chrono))} stale rev {st.rev.mean()*100:4.1f}"
    else:
        out += " CML not judged"
    pm = raw[raw.task == "personamem"]
    if len(pm):
        p = pm.pivot_table(index="id", columns="cond", values="correct", aggfunc="first").dropna().astype(float)
        out += f" | PM n={len(p)} {p.chrono.mean()*100:5.1f}/{p.rev.mean()*100:5.1f} {f(boot(p.rev-p.chrono))}"
    print(out)

print("\n#### Logs (string): newest_first - oldest_first; per format; k")
for m in MODELS:
    d = load(f"app_logs_{m}.jsonl")
    if d.empty:
        continue
    p = d.pivot_table(index=["seed", "fmt", "k"], columns="order", values="correct", aggfunc="first").astype(float)
    fm = d.groupby(["fmt", "order"]).correct.mean().unstack() * 100
    per = " ".join(f"{x}:{fm.loc[x,'oldest_first']:.0f}->{fm.loc[x,'newest_first']:.0f}" for x in fm.index)
    print(f"  {tag(m):12s} {p.oldest_first.mean()*100:5.1f}->{p.newest_first.mean()*100:5.1f} {f(boot(p.newest_first-p.oldest_first))} | {per}")

print("\n#### Agent: stale memory before the question (end) vs start, minus base")
for m in MODELS:
    d = load(f"app_agent_{m}.jsonl")
    if d.empty:
        continue
    p = d.pivot_table(index=["seed", "k"], columns="cond", values="correct", aggfunc="first").astype(float)
    print(f"  {tag(m):12s} base {p.base.mean()*100:5.1f} end {f(boot(p.stale_mem_end-p.base))} start {f(boot(p.stale_mem_start-p.base))}")

print("\n#### CoT (HF scoring): accuracy by condition, k=8 pick-last in shuffled")
for m in MODELS:
    mm = m.replace("~vllm", "")
    d = load(f"ext_cot_{mm}.jsonl")
    if d.empty:
        continue
    acc = (d.groupby("cond").correct.mean() * 100).round(1).to_dict()
    pl = d[d.cond == "shuf"].groupby("k").pick_last_presented.mean().mul(100).round(1).to_dict()
    print(f"  {tag(m):12s} " + " ".join(f"{c} {acc.get(c, float('nan')):.1f}" for c in
          ("full", "shuf", "shuf_step", "shuf_neutral", "rev", "rev_step", "rev_header")) + f" | shuf pick-last by k {pl}")

print("\n#### E15 (pre-registered verdict on newest-first input)")
for m in MODELS:
    d = load(f"e15_rule_{m}.jsonl")
    if d.empty:
        continue
    res = {}
    for (qt, od), h in d.groupby(["qtype", "order"]):
        res[(qt, od)] = boot((h.pos == "last").astype(float) - (h.pos == "first").astype(float))
    cur, ear, chk = res[("current", "reverse")], res[("earliest", "reverse")], res[("earliest", "chrono")]
    v = ("SEMANTIC" if cur[0] >= 20 and ear[0] <= -20 else "MECHANICAL" if cur[0] >= 20 and ear[0] >= 20 else "MIXED")
    acc = d.groupby(["qtype", "order"]).correct.mean().mul(100).round(1)
    print(f"  {tag(m):12s} cur-rev last-first {f(cur)} | earliest-rev {f(ear)} | earliest-chrono {f(chk)} -> {v} | "
          f"acc cur {acc[('current','chrono')]}/{acc[('current','reverse')]} ear {acc[('earliest','chrono')]}/{acc[('earliest','reverse')]}")

print("\n#### External sets")
for m in MODELS:
    out = f"  {tag(m):12s}"
    d = load(f"ext_mab_{m}.jsonl")
    if not d.empty:
        d["s"] = d.src.str.extract(r"_(sh|mh)_")[0]
        a = d.groupby(["s", "order"]).correct.mean().mul(100).round(1)
        out += f" MAB sh {a['sh']['original']}/{a['sh']['reversed']}/{a['sh']['shuffled']} mh {a['mh']['original']}/{a['mh']['reversed']}"
    d = load(f"ext_tot_{m}.jsonl")
    if not d.empty:
        d = d[~d.get("skipped", pd.Series(False, index=d.index)).fillna(False).astype(bool)]
        a = d.groupby("order").correct.mean().mul(100).round(1)
        out += f" | ToT sorted {a.get('sorted')} shuffle {a.get('shuffle')} (n {len(d)//2})"
    d = load(f"ext_tempreason_{m}.jsonl")
    if not d.empty:
        a = d.groupby("order").correct.mean().mul(100).round(1)
        out += f" | TR {a.get('sorted')}/{a.get('reversed')}/{a.get('shuffled')}"
    d = load(f"ext_babilong_{m}.jsonl")
    if not d.empty:
        a = d.groupby("task").correct.mean().mul(100).round(1)
        out += f" | bAbI {a.get('qa1')}/{a.get('qa2')}/{a.get('qa3')}"
    d = load(f"ext_gsm8k_{m}.jsonl")
    if not d.empty:
        out += f" | GSM8K {d.correct.mean()*100:.1f}"
    print(out)


# ================================================================ pre-registered scale rule (10-06)
print()





def load2(name):
    p = os.path.join(R, name)
    if not os.path.exists(p) or not os.path.getsize(p):
        return pd.DataFrame()
    return pd.DataFrame([json.loads(l) for l in open(p, encoding="utf-8") if l.strip()])


def boot2(d, groups=None, seed=0):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    if groups is None:
        m = d[rng.integers(0, len(d), (B, len(d)))].mean(1)
    else:
        g = pd.Series(list(groups)).astype("category").cat.codes.values
        k = g.max() + 1
        s, c = np.bincount(g, weights=d, minlength=k), np.bincount(g, minlength=k)
        idx = rng.integers(0, k, (B, k))
        m = s[idx].sum(1) / c[idx].sum(1)
    return d.mean() * 100, np.percentile(m, 2.5) * 100, np.percentile(m, 97.5) * 100


def f2(t):
    return f"{t[0]:+5.1f} [{t[1]:+5.1f}, {t[2]:+5.1f}]"


def verdict(own, inter):
    rep = "replicated" if own[0] <= -10 and own[2] < 0 else ("vanished" if abs(own[0]) < 5 and own[1] <= 0 <= own[2] else "not replicated")
    att = ("attenuated" if inter[0] >= 10 and inter[1] > 0 else
           "comparable" if abs(inter[0]) < 10 and -15 <= inter[1] and inter[2] <= 15 else "undetermined")
    return f"32B {rep}; vs 4B {att}"


def contrast(piv32, piv4, a, b, groups=None):
    ids = piv32.index.intersection(piv4.index)
    e32 = piv32.loc[ids, a] - piv32.loc[ids, b]
    e4 = piv4.loc[ids, a] - piv4.loc[ids, b]
    g = None if groups is None else groups(ids)
    own, own4, inter = boot2(e32, g), boot2(e4, g), boot2(e32 - e4, g)
    return len(ids), own4, own, inter


rows = []
# MemConflict (judged v1)
def mc(m):
    raw = load2(f"extmem_{m}.jsonl")
    raw = raw[raw.task == "memconf"]
    j = load2(f"extmem_{m}_judged.jsonl")
    j = j[j.task == "memconf"][["id", "cond", "label"]]
    d = raw.merge(j, on=["id", "cond"])
    return d.assign(c=(d.label == "A").astype(float)).pivot_table(index="id", columns="cond", values="c", aggfunc="first")
p32, p4 = mc("Qwen3-32B~vllm"), mc("Qwen3-4B~vllm")
for c in ("rev", "retr"):
    rows.append((f"MemConflict {c}", *contrast(p32, p4, c, "chrono", lambda ids: ids.str[:8])))
# LongMemEval (seen)
def lme(m):
    j = load2(f"app_memory_{m}_judged.jsonl")
    j = j[~j.skipped.astype(bool)]
    return j.assign(c=(j.label == "A").astype(float)).pivot_table(index="qid", columns="cond", values="c", aggfunc="first")
l32, l4 = lme("Qwen3-32B~vllm"), lme("Qwen3-4B~vllm")
rows.append(("LongMemEval S dated (seen)", *contrast(l32, l4, "S_rev_dated", "S_chrono_dated")))
rows.append(("LongMemEval S dated+header (seen)", *contrast(l32, l4, "S_rev_dated_header", "S_chrono_dated")))
# ConvoMem-long (judged)
def cml(m):
    j = load2(f"longconv_{m}_judged.jsonl")
    return j.assign(c=(j.label == "A").astype(float)).pivot_table(index="id", columns="cond", values="c", aggfunc="first")
c32, c4 = cml("Qwen3-32B~vllm"), cml("Qwen3-4B~vllm")
rows.append(("ConvoMem-long", *contrast(c32, c4, "rev", "chrono", lambda ids: pd.Index(ids).str.split("-").str[0])))
# Logs
def logs(m):
    d = load2(f"app_logs_{m}.jsonl")
    return d.pivot_table(index=["seed", "fmt", "k"], columns="order", values="correct", aggfunc="first").astype(float)
g32, g4 = logs("Qwen3-32B~vllm"), logs("Qwen3-4B~vllm")
rows.append(("Logs newest-first", *contrast(g32, g4, "newest_first", "oldest_first")))
# CoT (HF scoring, files without ~vllm)
def cot(m):
    d = load2(f"ext_cot_{m}.jsonl")
    return d.assign(key=d.k.astype(str) + "_" + d.i.astype(str)).pivot_table(index="key", columns="cond", values="correct", aggfunc="first").astype(float)
t32, t4 = cot("Qwen3-32B"), cot("Qwen3-4B")
for c in ("shuf", "rev_step", "shuf_step", "rev_header"):
    rows.append((f"CoT {c} vs full", *contrast(t32, t4, c, "full")))

print(f"{'test':34s} {'n':>4s}  {'4B effect':22s} {'32B effect':22s} {'32B-4B':22s} verdict")
for name, n, own4, own, inter in rows:
    print(f"{name:34s} {n:4d}  {f2(own4):22s} {f2(own):22s} {f2(inter):22s} {verdict(own, inter)}")

print("\n== MemConflict string rule (mc_preview: new value mentioned, old absent or before it) for every model")
def key(v):
    return str(v).split(",")[0].strip().lower()
for m in ["Qwen3-4B~vllm", "Qwen3-8B-bf16~vllm", "Qwen3-14B-bf16~vllm", "Qwen3-32B~vllm", "Gemma-3-12B~vllm"]:
    raw = load2(f"extmem_{m}.jsonl")
    raw = raw[raw.task == "memconf"].copy()
    t = raw.response.astype(str).str.lower()
    o = np.array([x.find(key(v)) for x, v in zip(t, raw.old)])
    nn = np.array([x.find(key(v)) for x, v in zip(t, raw.new)])
    raw["new_like"] = ((nn >= 0) & ((o < 0) | (o < nn))).astype(float)
    raw["old_like"] = ((o >= 0) & ((nn < 0) | (nn < o))).astype(float)
    p = raw.pivot_table(index="id", columns="cond", values="new_like", aggfunc="first")
    s = raw.pivot_table(index="id", columns="cond", values="old_like", aggfunc="first")
    cl = p.index.str[:8]
    print(f"  {m:22s} n={len(p)} new {p.chrono.mean()*100:5.1f}/{p.rev.mean()*100:5.1f}/{p.retr.mean()*100:5.1f} "
          f"old {s.chrono.mean()*100:4.1f}/{s.rev.mean()*100:4.1f}/{s.retr.mean()*100:4.1f} | rev {f2(boot2(p.rev-p.chrono))} "
          f"cl {f2(boot2(p.rev-p.chrono, cl))} | retr {f2(boot2(p.retr-p.chrono))}")
