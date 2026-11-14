"""Cross-harness compaction comparison (PREREG_COMPACTION_HARNESS.md).

python swechat_harness.py extract  -> per-session records for every harness with compaction
python swechat_harness.py measure  -> measure 1 (verbatim retention) and measure 2 (pushback change)
python swechat_harness.py sample   -> measure 3 sample (new harnesses), written for blind labelling
"""
import json, random, re, sys, collections
import pyarrow as pa, pyarrow.compute as pc
from pyarrow.parquet import ParquetFile

SRC = r"D:\hf_cache\manual\swe-chat\conversations.parquet"
W = "D:/hf_cache/manual/swe-chat-work/"
MARK = "This session is being continued from a previous conversation"
PUSH = {"correction", "failure_report", "rejection"}
MARKERS = {"other:codex:compacted": "Codex", "other:pi:compaction": "Pi", "other:opencode:compaction": "OpenCode",
           "other:session.compaction_complete": "Copilot CLI", "other:context.apply_compaction": "Kimi Code"}
GROUP = {"Claude Code": "verbatim", "Codex": "verbatim", "Pi": "free", "Oh My Pi": "free",
         "OpenCode": "free", "Copilot CLI": "free", "Kimi Code": "other"}
COLS = ["session_id", "repo_id", "agent", "role", "turn_type", "turn_number", "content_block_index",
        "content", "prompt_pushback", "source_entry_uuid"]
norm = lambda t: re.sub(r"\s+", "", t or "")


def is_comp(r):
    if r["agent"] == "Claude Code":
        return r["role"] == "user" and MARK in (r["content"] or "")[:400]
    return r["turn_type"] in MARKERS


def labelled(r):
    return (r["role"] == "user" and r["turn_type"] == "user_prompt" and r["prompt_pushback"] is not None
            and MARK not in (r["content"] or "")[:400])


def context(rows, k):
    """Text kept after the compaction at index k, plus index of the first verbatim-kept row (Pi) or None."""
    r = rows[k]
    if r["agent"] == "Claude Code":
        return r["content"], None
    d = json.loads(r["content"])
    t = r["turn_type"]
    if t == "other:codex:compacted":
        parts = []
        for x in d["payload"].get("replacement_history") or []:
            for c in x.get("content") or []:
                if isinstance(c, dict):
                    parts.append(c.get("text") or "")
        return "\n".join(parts), None
    if t == "other:pi:compaction":
        kept = next((i for i in range(k) if rows[i]["source_entry_uuid"] == d.get("firstKeptEntryId")), None)
        return d.get("summary") or "", kept if kept is not None else -1
    if t == "other:session.compaction_complete":
        return d["data"].get("summaryContent") or "", None
    if t == "other:opencode:compaction":
        for i in range(k + 1, min(k + 6, len(rows))):
            if rows[i]["role"] == "assistant" and rows[i]["content"]:
                return rows[i]["content"], None
        return "", None
    return d.get("summary") or "", None


def harness(r):
    if r["agent"] == "Claude Code":
        return "Claude Code"
    return r["agent"] if r["agent"] in ("Oh My Pi",) else MARKERS[r["turn_type"]]


def extract():
    pf = ParquetFile(SRC)
    cc = json.load(open(W + "compaction_sessions.json"))
    ids = {s for s, v in cc.items() if "S" in v["seq"]}
    for b in pf.iter_batches(columns=["session_id", "turn_type", "agent"]):
        m = pc.and_(pc.is_in(b["turn_type"], value_set=pa.array(sorted(MARKERS))), pc.not_equal(b["agent"], "Claude Code"))
        ids |= set(b.filter(m)["session_id"].to_pylist())
    sel = pa.array(sorted(ids))
    rows = collections.defaultdict(list)
    for b in pf.iter_batches(columns=COLS):
        for r in b.filter(pc.is_in(b["session_id"], value_set=sel)).to_pylist():
            rows[r["session_id"]].append(r)
    out, texts = {}, {}
    for s, rs in rows.items():
        rs.sort(key=lambda r: (r["turn_number"] or 0, r["content_block_index"] or 0))
        comps = [i for i, r in enumerate(rs) if is_comp(r)]
        if not comps:
            continue
        k = comps[0]
        h = harness(rs[k])
        end = comps[1] if len(comps) > 1 else len(rs)
        ctx, kept = context(rs, k)
        nctx = norm(ctx)
        pre = [i for i in range(k) if labelled(rs[i])]
        post = [i for i in range(k + 1, end) if labelled(rs[i])]
        ret = []
        for i in pre:
            t = rs[i]["content"] or ""
            if len(t.strip()) >= 20 and not t.lstrip().startswith("<"):
                ret.append(int(norm(t)[:60] in nctx or (kept is not None and kept >= 0 and i >= kept)))
        out[s] = {"harness": h, "repo": rs[k]["repo_id"], "n_comp": len(comps), "pi_kept_found": kept,
                  "pre": "".join("P" if rs[i]["prompt_pushback"] in PUSH else "N" for i in pre),
                  "post": "".join("P" if rs[i]["prompt_pushback"] in PUSH else "N" for i in post), "ret": ret}
        texts[s] = {"ctx": ctx, "pre": [rs[i]["content"] for i in pre], "post": [rs[i]["content"] for i in post],
                    "typed_before": [rs[i]["content"] for i in range(end) if rs[i]["role"] == "user"
                                     and rs[i]["turn_type"] == "user_prompt" and MARK not in (rs[i]["content"] or "")[:400]],
                    "pre_idx": pre, "post_idx": post,
                    "typed_idx": [i for i in range(end) if rs[i]["role"] == "user" and rs[i]["turn_type"] == "user_prompt"
                                  and MARK not in (rs[i]["content"] or "")[:400]]}
    json.dump(out, open(W + "harness_sessions.json", "w"))
    json.dump(texts, open(W + "harness_texts.json", "w", encoding="utf-8"), ensure_ascii=False)
    print(collections.Counter(v["harness"] for v in out.values()))


def boot(items, reps=2000, seed=0):
    by = collections.defaultdict(list)
    for r, v in items:
        by[r].append(v)
    repos = sorted(by)
    rng = random.Random(seed)
    ms = []
    for _ in range(reps):
        vals = [v for r in (rng.choice(repos) for _ in repos) for v in by[r]]
        ms.append(sum(vals) / len(vals))
    ms.sort()
    return sum(v for _, v in items) / len(items), ms[int(.025 * reps)], ms[int(.975 * reps)]


def rate(x):
    return x.count("P") / len(x)


def measure():
    S = json.load(open(W + "harness_sessions.json"))
    hs = sorted({v["harness"] for v in S.values()}, key=lambda h: -sum(v["harness"] == h for v in S.values()))
    print("== measure 1: verbatim retention of pre-compaction user messages (first compaction per session)")
    for h in hs:
        items = [(s, x) for s, v in S.items() if v["harness"] == h for x in v["ret"]]
        nsess = sum(v["harness"] == h for v in S.values())
        extra = ""
        if h in ("Pi", "Oh My Pi"):
            extra = f" | kept-id matched {sum(1 for v in S.values() if v['harness'] == h and (v['pi_kept_found'] or -1) >= 0)}/{nsess}"
        if items:
            m, lo, hi = boot(items)
            print(f"  {h}: sessions {nsess}, messages {len(items)}, retained {100*m:.1f}% [{100*lo:.1f},{100*hi:.1f}]{extra}")
        else:
            print(f"  {h}: sessions {nsess}, no qualifying messages")
    print("== measure 2: pushback change, window 3")
    res = {}
    for h in hs + ["verbatim", "free"]:
        prim, plac, pre_r, post_r = [], [], [], []
        for s, v in S.items():
            if h not in (v["harness"], GROUP[v["harness"]]):
                continue
            b, a = v["pre"], v["post"]
            if len(b) >= 3 and len(a) >= 3:
                prim.append((v["repo"], rate(a[:3]) - rate(b[-3:])))
                pre_r.append(rate(b[-3:])); post_r.append(rate(a[:3]))
                if len(b) >= 6:
                    plac.append((v["repo"], rate(b[-3:]) - rate(b[-6:-3])))
        if not prim:
            continue
        m, lo, hi = boot(prim)
        res[h] = prim
        line = (f"  {h}: n={len(prim)} repos={len({r for r, _ in prim})} pre {100*sum(pre_r)/len(pre_r):.1f}% "
                f"post {100*sum(post_r)/len(post_r):.1f}% diff {100*m:+.1f} [{100*lo:+.1f},{100*hi:+.1f}]")
        if plac:
            pm, plo, phi = boot(plac)
            line += f" | placebo n={len(plac)} {100*pm:+.1f} [{100*plo:+.1f},{100*phi:+.1f}]"
        print(line)
    # primary contrast: free minus verbatim, bootstrapping repos within each group independently
    rng = random.Random(0)
    def grp(items):
        by = collections.defaultdict(list)
        for r, v in items:
            by[r].append(v)
        return by
    F, V = grp(res["free"]), grp(res["verbatim"])
    def draw(by):
        repos = sorted(by)
        vals = [x for r in (rng.choice(repos) for _ in repos) for x in by[r]]
        return sum(vals) / len(vals)
    d = sorted(draw(F) - draw(V) for _ in range(2000))
    point = sum(v for _, v in res["free"]) / len(res["free"]) - sum(v for _, v in res["verbatim"]) / len(res["verbatim"])
    print(f"== primary contrast free - verbatim: {100*point:+.1f}pp [{100*d[50]:+.1f},{100*d[1949]:+.1f}]")


def sample():
    S = json.load(open(W + "harness_sessions.json"))
    T = json.load(open(W + "harness_texts.json", encoding="utf-8"))
    plan = {("free", "post"): 80, ("free", "pre"): 40, ("Codex", "post"): 60, ("Codex", "pre"): 30}
    pools = collections.defaultdict(list)
    for s in sorted(S):
        v = S[s]
        g = "free" if GROUP[v["harness"]] == "free" else v["harness"]
        if g not in ("free", "Codex") or len(v["pre"]) < 3 or len(v["post"]) < 3:
            continue
        for j, c in enumerate(v["post"][:3]):
            if c == "P":
                pools[g, "post"].append((s, v["repo"], "post", j))
        for j, c in enumerate(v["pre"][-3:]):
            if c == "P":
                pools[g, "pre"].append((s, v["repo"], "pre", len(v["pre"]) - 3 + j))
    items = []
    for key, n in plan.items():
        pool = pools[key][:]
        random.Random(0).shuffle(pool)
        cnt, picked = collections.Counter(), []
        for it in pool:
            if cnt[it[1]] < 2:
                cnt[it[1]] += 1
                picked.append(it)
            if len(picked) == n:
                break
        print(key, "eligible", len(pool), "picked", len(picked), "repos", len(cnt))
        for s, repo, grp_, j in picked:
            t = T[s]
            row = (t["post_idx"] if grp_ == "post" else t["pre_idx"])[j]
            earlier = [txt for i, txt in zip(t["typed_idx"], t["typed_before"]) if i < row and not (txt or "").startswith("<")]
            items.append({"id": f"{key[0][:2]}{grp_[:2]}{len(items):03d}", "harness": S[s]["harness"], "group": grp_,
                          "session": s, "repo": repo, "text": (t["post"] if grp_ == "post" else t["pre"])[j],
                          "earlier": earlier, "ctx": t["ctx"]})
    json.dump(items, open(W + "harness_sample.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("written", len(items))


if __name__ == "__main__":
    {"extract": extract, "measure": measure, "sample": sample}[sys.argv[1]]()
