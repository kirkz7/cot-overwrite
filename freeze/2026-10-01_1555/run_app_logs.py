"""Application 2: newest-first logs. Does the reader report the CURRENT value of a setting when the
history is shown newest-first (the default for `git log`, Keep-a-Changelog files and many email
views), even though every entry carries a date?

Formats: git log, CHANGELOG.md, email thread. Orders: oldest-first vs newest-first.
k = number of changes to the queried setting after its initial value (1, 2, 4); 10 entries total.
usage: python run_app_logs.py --models Qwen3-4B,Qwen3-14B --n 100
"""
import argparse
import datetime as dt
import random

import pandas as pd
from tqdm import tqdm

from app_common import tag, JsonlAppender, answer_fast, chat_prompt, first_number, free_gpu, load_reader

KEYS = [("request_timeout", "s", 10, 180), ("max_connections", "", 20, 900), ("cache_ttl", "s", 60, 3600),
        ("batch_size", "", 8, 512), ("retry_limit", "", 2, 40), ("worker_count", "", 2, 96),
        ("upload_limit", " MB", 5, 900), ("session_timeout", " min", 5, 240), ("rate_limit", " req/min", 50, 5000),
        ("queue_depth", "", 16, 2048)]
NAMES = ["Alice Chen", "Bob Martin", "Chen Wei", "Dana Ortiz", "Eli Novak", "Fatima Khan", "Gus Petrov", "Hana Sato"]
NOISE = {"git": ["Fix typo in README", "Add unit tests for the request parser", "Refactor logging setup",
                 "Update CI workflow to Python 3.12", "Remove unused imports"],
         "changelog": ["Crash when the config file is empty.", "Incorrect retry count in error messages.",
                       "Typo in the CLI help text.", "Memory leak in the metrics exporter."],
         "email": ["Thanks, I'll review the PR tomorrow.", "Reminder: the release sync moved to Thursday.",
                   "Can someone check why the nightly build failed?", "Looks good to me."]}
KS = [1, 2, 4]
N_ENTRIES = 10


def make_item(k, seed):
    rng = random.Random(seed)
    keys = rng.sample(KEYS, 3)
    target, others = keys[0], keys[1:]
    used = set()

    def val(key):
        while True:
            v = rng.randint(key[2], key[3])
            if v not in used:
                used.add(v)
                return v
    entries = [dict(key=target[0], unit=target[1], value=val(target), kind="set") for _ in range(k + 1)]
    n_other = rng.randint(2, 3)
    entries += [dict(key=o[0], unit=o[1], value=val(o), kind="set") for o in rng.choices(others, k=n_other)]
    entries += [dict(kind="noise") for _ in range(N_ENTRIES - len(entries))]
    # random chronological order, but the target's own updates keep their relative order (they are the history)
    t_entries = [e for e in entries if e.get("key") == target[0]]
    rest = [e for e in entries if e.get("key") != target[0]]
    rng.shuffle(rest)
    slots = sorted(rng.sample(range(N_ENTRIES), k + 1))
    chrono, ti, ri = [], 0, 0
    for i in range(N_ENTRIES):
        if i in slots:
            chrono.append(t_entries[ti]); ti += 1
        else:
            chrono.append(rest[ri]); ri += 1
    t = dt.datetime(2025, rng.randint(1, 6), rng.randint(1, 28), rng.randint(8, 18), rng.randint(0, 59))
    for i, e in enumerate(chrono):
        e["date"] = t
        e["author"] = rng.choice(NAMES)
        e["hash"] = "%07x" % rng.getrandbits(28)
        e["version"] = "1.%d.0" % (i + 3)
        e["noise_idx"] = rng.randrange(5)
        t += dt.timedelta(days=rng.randint(1, 9), hours=rng.randint(0, 7), minutes=rng.randint(0, 59))
    history = [e["value"] for e in chrono if e.get("key") == target[0]]
    return dict(k=k, seed=seed, target=target[0], unit=target[1], entries=chrono, history=history)


def render(item, fmt, order):
    es = item["entries"] if order == "oldest_first" else item["entries"][::-1]
    blocks = []
    for e in es:
        d = e["date"]
        if fmt == "git":
            msg = f"Set {e['key']} to {e['value']}{e['unit']}" if e["kind"] == "set" else NOISE["git"][e["noise_idx"] % 5]
            email = e["author"].split()[0].lower() + "@example.com"
            blocks.append(f"commit {e['hash']}\nAuthor: {e['author']} <{email}>\nDate:   {d.strftime('%a %b %d %H:%M:%S %Y')} +0800\n\n    {msg}\n")
        elif fmt == "changelog":
            body = (f"### Changed\n- `{e['key']}` is now {e['value']}{e['unit']}." if e["kind"] == "set"
                    else f"### Fixed\n- {NOISE['changelog'][e['noise_idx'] % 4]}")
            blocks.append(f"## [{e['version']}] - {d.strftime('%Y-%m-%d')}\n{body}\n")
        else:
            body = (f"Quick update: {e['key']} is now {e['value']}{e['unit']}." if e["kind"] == "set"
                    else NOISE["email"][e["noise_idx"] % 4])
            blocks.append(f"From: {e['author']}\nDate: {d.strftime('%a, %d %b %Y %H:%M')}\nSubject: Re: service settings\n\n{body}\n")
    if fmt == "changelog":
        doc = "# Changelog\n\nAll notable changes to this project are documented in this file.\n\n" + "\n".join(blocks)
        q = f"According to the changelog above, what is the current value of {item['target']}?"
    elif fmt == "git":
        doc = "$ git log\n" + "\n".join(blocks) if order == "newest_first" else "$ git log --reverse\n" + "\n".join(blocks)
        q = f"According to the commit history above, what is the current value of {item['target']}?"
    else:
        doc = "\n---\n".join(blocks)
        q = f"According to the email thread above, what is the current value of {item['target']}?"
    shown = [e["value"] for e in es if e.get("key") == item["target"]]
    return doc + "\n\n" + q + " Answer with the number only.", shown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default="Qwen3-4B,Qwen3-14B,OLMo-2-13B-Instruct,Phi-4-mini")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--show", action="store_true")
    args = ap.parse_args()
    items = [make_item(k, 7_000_000 + k * 10_000 + i) for k in KS for i in range(args.n)]
    if args.show:
        for fmt in ("git", "changelog", "email"):
            p, shown = render(items[args.n], fmt, "newest_first")
            print("=" * 30, fmt, "| history", items[args.n]["history"], "| shown", shown, "\n", p)
        return
    jobs = [(it, fmt, order) for it in items for fmt in ("git", "changelog", "email") for order in ("oldest_first", "newest_first")]
    for name in args.models.split(","):
        out = f"results/app_logs_{tag(name)}.jsonl"
        w = JsonlAppender(out, key=lambda r: (r["seed"], r["fmt"], r["order"]))
        todo = [j for j in jobs if (j[0]["seed"], j[1], j[2]) not in w.done]
        print(name, "done", len(jobs) - len(todo), "todo", len(todo), flush=True)
        if todo:
            tok, model = load_reader(name)
            for it, fmt, order in tqdm(todo, desc=name):
                user, shown = render(it, fmt, order)
                raw = answer_fast(tok, model, chat_prompt(tok, user))
                pred = first_number(raw)
                h = it["history"]
                w.write(dict(model=name, k=it["k"], seed=it["seed"], fmt=fmt, order=order, raw=raw,
                             pred=pred, correct=pred == h[-1], stale=pred in h[:-1],
                             pick_last_shown=pred == shown[-1], pick_first_shown=pred == shown[0]))
            del tok, model
            free_gpu()
        w.close()
        df = pd.DataFrame(w.rows)
        print(name, "\n", (df.groupby(["fmt", "order", "k"]).correct.mean().unstack("k") * 100).round(1).to_string(), flush=True)


if __name__ == "__main__":
    main()
