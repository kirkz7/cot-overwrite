"""Gap 1 (PREREG_COMPACTION_REAL.md): user pushback around real Claude Code compactions in SWE-chat.

python swechat_compaction.py extract   -> per-session user-turn sequences (no text)
python swechat_compaction.py part1     -> full deterministic comparison with controls A and B
python swechat_compaction.py sample    -> part 2 sample (60 + 60) with the text needed for hand labelling
"""
import json, random, sys, collections
import pyarrow.parquet as pq

SRC = r"D:\hf_cache\manual\swe-chat\conversations.parquet"
W = "D:/hf_cache/manual/swe-chat-work/"
MARK = "This session is being continued from a previous conversation"
PUSH = {"correction", "failure_report", "rejection"}
COLS = ["session_id", "repo_id", "agent", "role", "turn_type", "turn_number", "content_block_index", "content", "prompt_pushback"]


def extract():
    f = pq.ParquetFile(SRC)
    sess = collections.defaultdict(list)
    repo = {}
    for i in range(f.num_row_groups):
        for r in f.read_row_group(i, columns=COLS).to_pylist():
            if r["agent"] != "Claude Code" or r["role"] != "user":
                continue
            c = r["content"] or ""
            if MARK in c[:400]:
                kind = "S"
            elif "<command-name>/compact" in c:
                kind = "C"
            elif r["turn_type"] == "user_prompt" and r["prompt_pushback"] is not None:
                kind = "P" if r["prompt_pushback"] in PUSH else "N"
            else:
                continue
            sess[r["session_id"]].append((r["turn_number"] or 0, r["content_block_index"] or 0, kind))
            repo[r["session_id"]] = r["repo_id"]
    out = {s: {"repo": repo[s], "seq": "".join(k for _, _, k in sorted(v))} for s, v in sess.items()}
    json.dump(out, open(W + "compaction_sessions.json", "w"))
    print("sessions", len(out), "| with summary", sum("S" in v["seq"] for v in out.values()))


def split(seq):
    """Labelled turns before the first summary, labelled turns until the next summary, manual flag."""
    k = seq.index("S")
    before = [c for c in seq[:k] if c in "PN"]
    after = [c for c in seq[k + 1:].split("S")[0] if c in "PN"]
    manual = "C" in seq[max(0, k - 3):k]
    return before, after, manual


def rate(xs):
    return sum(c == "P" for c in xs) / len(xs)


def boot(items, reps=2000):
    """items: list of (repo, value); resample whole repos."""
    by = collections.defaultdict(list)
    for r, v in items:
        by[r].append(v)
    repos = sorted(by)
    rng = random.Random(0)
    ms = []
    for _ in range(reps):
        vals = [v for r in (rng.choice(repos) for _ in repos) for v in by[r]]
        ms.append(sum(vals) / len(vals))
    ms.sort()
    return sum(v for _, v in items) / len(items), ms[int(.025 * reps)], ms[int(.975 * reps)]


def fmt(name, items):
    m, lo, hi = boot(items)
    return f"{name}: n={len(items)} repos={len({r for r, _ in items})} diff={100*m:+.1f}pp CI[{100*lo:+.1f},{100*hi:+.1f}]"


def part1():
    S = json.load(open(W + "compaction_sessions.json"))
    comp = {s: v for s, v in S.items() if "S" in v["seq"]}
    res = {}
    for k in (3, 1, 5):
        prim, prim_a, plac, manual, auto, bvals = [], [], [], [], [], []
        pre_rates, post_rates = [], []
        for s, v in comp.items():
            b, a, man = split(v["seq"])
            if len(b) >= k and len(a) >= k:
                d = rate(a[:k]) - rate(b[-k:])
                prim.append((v["repo"], d)); bvals.append(len(b))
                pre_rates.append(rate(b[-k:])); post_rates.append(rate(a[:k]))
                (manual if man else auto).append((v["repo"], d))
                if len(b) >= 2 * k:
                    plac.append((v["repo"], rate(b[-k:]) - rate(b[-2 * k:-k])))
                    prim_a.append((v["repo"], d))
        rng = random.Random(0)
        ctrl = []
        for s in sorted(S):
            v = S[s]
            if "S" in v["seq"]:
                continue
            L = [c for c in v["seq"] if c in "PN"]
            ok = [x for x in bvals if k <= x <= len(L) - k]
            if len(L) >= 2 * k and ok:
                x = rng.choice(ok)
                ctrl.append((v["repo"], rate(L[x:x + k]) - rate(L[x - k:x])))
        print(f"--- window {k}: pre rate {100*sum(pre_rates)/len(pre_rates):.1f}% post rate {100*sum(post_rates)/len(post_rates):.1f}%")
        for name, it in (("primary", prim), ("primary on control-A subset", prim_a), ("control A placebo", plac),
                         ("control B no-compaction", ctrl), ("manual /compact", manual), ("auto", auto)):
            if it:
                print("  " + fmt(name, it))
                res[f"k{k}:{name}"] = boot(it) + (len(it),)
    json.dump(res, open("results/compaction_part1.json", "w"), indent=1)


def sample():
    S = json.load(open(W + "compaction_sessions.json"))
    groups = {"post": [], "pre": []}
    for s in sorted(S):
        v = S[s]
        if "S" not in v["seq"]:
            continue
        b, a, _ = split(v["seq"])
        if len(b) >= 3 and len(a) >= 3:
            for j, c in enumerate(a[:3]):
                if c == "P":
                    groups["post"].append((s, v["repo"], j))
            for j, c in enumerate(b[-3:]):
                if c == "P":
                    groups["pre"].append((s, v["repo"], len(b) - 3 + j))
    picked = {}
    for g, items in groups.items():
        rng = random.Random(0)
        rng.shuffle(items)
        cnt, out = collections.Counter(), []
        for it in items:
            if cnt[it[1]] < 2:
                cnt[it[1]] += 1
                out.append(it)
            if len(out) == 60:
                break
        picked[g] = out
        print(g, "eligible", len(items), "| picked", len(out), "| repos", len(cnt))
    need = {s for g in picked.values() for s, _, _ in g}
    f = pq.ParquetFile(SRC)
    rows = collections.defaultdict(list)
    for i in range(f.num_row_groups):
        for r in f.read_row_group(i, columns=COLS).to_pylist():
            if r["session_id"] in need and r["role"] == "user":
                rows[r["session_id"]].append(r)
    items = []
    for g, lst in picked.items():
        for n, (s, repo, j) in enumerate(lst, 1):
            rs = sorted(rows[s], key=lambda r: (r["turn_number"] or 0, r["content_block_index"] or 0))
            k = next(i for i, r in enumerate(rs) if MARK in (r["content"] or "")[:400])
            lab = [i for i, r in enumerate(rs[:k]) if r["turn_type"] == "user_prompt" and r["prompt_pushback"] is not None
                   and MARK not in (r["content"] or "")[:400]]
            nxt = [i for i, r in enumerate(rs) if i > k and MARK in (r["content"] or "")[:400]]
            end = nxt[0] if nxt else len(rs)
            lab_after = [i for i in range(k + 1, end) if rs[i]["turn_type"] == "user_prompt" and rs[i]["prompt_pushback"] is not None]
            tgt = lab_after[j] if g == "post" else lab[j]
            earlier = [rs[i]["content"] for i in range(tgt) if rs[i]["turn_type"] == "user_prompt"
                       and MARK not in (rs[i]["content"] or "")[:400] and not (rs[i]["content"] or "").startswith("<")]
            items.append({"id": f"{g[:2].upper()}{n:02d}", "group": g, "session": s, "repo": repo,
                          "label": rs[tgt]["prompt_pushback"], "text": rs[tgt]["content"],
                          "earlier": earlier, "summary": rs[k]["content"]})
    json.dump(items, open(W + "compaction_sample.json", "w", encoding="utf-8"), ensure_ascii=False)
    print("written", len(items))


if __name__ == "__main__":
    {"extract": extract, "part1": part1, "sample": sample}[sys.argv[1]]()
