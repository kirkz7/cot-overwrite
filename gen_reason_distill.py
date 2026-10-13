"""E19.2 (EXPLORE_PLAN.md, 10-12): E19's training data with ONE change - the one-line thoughts of the "reason" and
"absent" items (thinking on) are replaced by the BASE model's own thoughts on those same prompts (self-distillation),
instead of E19.1's dated list. E19.1 removed the one-line "decide without reading" shortcut (MemConflict +7.9 over E19)
but pushed the list onto PersonaMem's reason / evolution questions (65.5 vs base 71.4 / 73.9); here the shortcut is
removed without teaching the list on questions that do not ask for a dated value.

run:      Qwen3-4B (no adapter), thinking on, Qwen3's recommended sampling, seeded per item and attempt; up to 4 attempts
          per item; an attempt is kept when the thought is closed and the visible reply is right (reason: the reason
          phrase or >= 60% of its content words; absent: says the attribute is not mentioned). Resumable.
assemble: bind7 = bind6 (E19.1, same rows / order / finals) with those rows' thoughts replaced by the first kept attempt
          (our own final answer is kept, so the answer format is unchanged); an item with no kept attempt keeps its
          E19.1 list-first thought (counted). val / dev rows stay as in bind6 (loss monitoring only).
usage: python gen_reason_distill.py run | assemble
"""
import argparse
import collections
import hashlib
import json
import re

from app_common import JsonlAppender, chat_prompt, load_jsonl, strip_think

RAW = "data_train/reason_distill_raw.jsonl"
KINDS = ("reason", "absent", "chat_reason", "chat_absent")
ATTEMPTS, MAX_NEW = 4, 2048
STOP = {"because", "about", "after", "before", "their", "there", "which", "would", "could", "since", "with", "from", "that",
        "this", "these", "those", "were", "been", "have", "kept", "being", "into", "over", "under"}
ABSENT = re.compile(r"\bnot (?:been )?(?:mention|specif|provid|stat|record|includ|given|available|found|list|disclos|indicat)"
                    r"|\bno (?:mention|information|record|data|details|reference)|n't (?:mention|say|specify|provide|include)"
                    r"|does not (?:mention|say|specify|provide|include|appear)|\bunknown\b|cannot be determined|not known", re.I)


def targets():
    return [r for r in load_jsonl("data_train/bind6_train.jsonl") if r["kind"] in KINDS and r["think"]]


def reason_of(final):
    return final.rsplit("because ", 1)[1].rstrip(". ")


def words(s):
    return {w for w in re.findall(r"[a-z]+", s.lower()) if len(w) >= 4 and w not in STOP}


def correct(row, visible):
    if row["kind"].endswith("absent"):
        return bool(ABSENT.search(visible))
    r = reason_of(row["final"])
    if r.lower() in visible.lower():
        return True
    w = words(r)
    return bool(w) and len(w & words(visible)) / len(w) >= 0.6


def run():
    import torch
    from app_common import load_reader
    from explore_probe import LONG_SDPA, sdpa_kernel      # memory-safe attention for long prompts (NOTEBOOK 10)
    from run_app_fix import generate_think
    rows = targets()
    w = JsonlAppender(RAW, key=lambda r: (r["id"], r["attempt"]))
    ok = {r["id"] for r in w.rows if r["ok"]}
    tok, model = load_reader("Qwen3-4B")
    from tqdm import tqdm
    for row in tqdm(rows, desc="distill"):
        for a in range(ATTEMPTS):
            if row["id"] in ok:
                break
            if (row["id"], a) in w.done:
                continue
            p = chat_prompt(tok, row["prompt"], think=True)
            with sdpa_kernel(LONG_SDPA, set_priority=True):   # 10-12: missing at first -> spilled into shared memory
                text = generate_think(tok, model, p, MAX_NEW, f"{row['id']}#{a}")
            full = text if text.startswith("<think>") else "<think>\n" + text
            fin = "</think>" in full
            vis = strip_think(full) if fin else ""
            good = fin and correct(row, vis)
            thought = full.split("</think>", 1)[0].removeprefix("<think>").strip() if fin else ""
            w.write(dict(id=row["id"], kind=row["kind"], attempt=a, finished=fin, ok=good, thought=thought, visible=vis,
                         n_tok=len(tok(full, add_special_tokens=False).input_ids)))
            if good:
                ok.add(row["id"])
            torch.cuda.empty_cache()
    w.close()


def assemble():
    raw = load_jsonl(RAW)
    first = {}
    for r in sorted(raw, key=lambda r: r["attempt"]):
        if r["ok"] and r["id"] not in first:
            first[r["id"]] = r
    want = {r["id"] for r in targets()}
    # 10-12: refuse to assemble from an unfinished run (a crashed `run` once let the queue go on to training)
    tried = collections.Counter(r["id"] for r in raw)
    todo = [i for i in want if i not in first and tried[i] < ATTEMPTS]
    assert not todo, f"distillation not finished: {len(todo)} of {len(want)} items still to try"
    stat = collections.Counter()
    for split in ("train", "val", "dev"):
        rows = load_jsonl(f"data_train/bind6_{split}.jsonl")
        if split == "train":
            for r in rows:
                # ids repeat across the E12 rows and the bind rows: match the kind as well (10-12 fix)
                if r["id"] in want and r["kind"] in KINDS and r["think"]:
                    if r["id"] in first:
                        r["answer"] = f"<think>\n{first[r['id']]['thought']}\n</think>\n\n{r['final']}"
                        stat[(r["kind"], "base thought")] += 1
                    else:
                        stat[(r["kind"], "kept list-first (no correct attempt)")] += 1
        path = f"data_train/bind7_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
        print(path, len(rows), hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16].upper())
    tried = collections.Counter(r["id"] for r in raw)
    print("items", len(want), "| attempts", len(raw), "| per-item attempts", dict(collections.Counter(tried.values())))
    for k, v in sorted(stat.items()):
        print(" ", k, v)
    kept = [first[i] for i in want if i in first]
    if kept:
        n = sorted(r["n_tok"] for r in kept)
        print("kept thought tokens: median", n[len(n) // 2], "| max", n[-1],
              "| contain 'from oldest to newest':", sum("from oldest to newest" in r["thought"] for r in kept))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("stage", choices=["run", "assemble"])
    {"run": run, "assemble": assemble}[ap.parse_args().stage]()
