"""E19.1 (EXPLORE_PLAN.md, 10-10): E19's training data with ONE change - every thought on a memory question starts with
the dated record list, including the "reason" and "absent" kinds, whose E18.1 / E19 thoughts were one line
("No record mentions X." / "The record from D ... says it was because R."). On MemConflict, E19 and E18.1 used such
one-line thoughts in half of the answers (accuracy 62% vs 80-89% with the list): a learned "decide without reading"
shortcut (NOTEBOOK 4i).

The bind4 rows are regenerated with gen_bind_data_v4's exact seeds (to recover each sample's dated chain) and checked
to be identical to data_train/bind4_decoupled_*.jsonl except the thought; the chat rows come from
`gen_chat_data.py --list_first`. Assembled in the same order as E19's bind5 (same shuffle seed), and the script first
rebuilds bind5 itself to check its hash.
usage: python gen_chat_data.py --list_first; python gen_bind_data_lf.py   -> data_train/bind6_{train,val,dev}.jsonl
"""
import collections
import copy
import hashlib
import json
import random

import gen_train_data as g
from gen_bind_data import make_sample, TRAIN_GENRES, DEV_GENRES
from gen_bind_data_v3 import think_answer
from gen_bind_data_v4 import build

BIND5_TRAIN = "9E4BAABF94B35CB3"


def sha(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16].upper()


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8")]


def records(a, chain):
    return "\n".join([f"Records about {a}, from oldest to newest:"] + [f"- {d}: {x}" for d, x in chain])


def bind4_lf(split, n, seed, genres, wiki):
    """bind4 rows of this split (no E12 / general rows), thoughts list-first; dict id -> row"""
    rng = random.Random(seed)
    out = {}
    for i in range(n):
        s = make_sample(rng, wiki, "decoupled", i, genres)
        v = copy.deepcopy(s["_v2"])
        r2 = random.Random(f"e181-{split}-{i}")
        row, thought = build(s, split, r2)
        row.pop("_v2", None)
        row["think"] = r2.random() < 0.5
        if row["kind"] == "reason":
            thought = records(v["a"], v["chain"]) + "\n" + thought
        elif row["kind"] == "absent":
            a2n = thought.removeprefix("No record mentions ").removesuffix(".")
            thought = f"Records about {a2n}, from oldest to newest:\n- none\n" + thought
        row["answer"] = think_answer(thought, row["final"], row["think"])
        out[row["id"]] = row
    return out


def main():
    wiki = g.load_wiki()
    for split, n, seed, genres in (("train", 2500, 41, TRAIN_GENRES), ("val", 300, 42, TRAIN_GENRES), ("dev", 300, 43, DEV_GENRES)):
        old4 = load(f"data_train/bind4_decoupled_{split}.jsonl")
        chat, chat_lf = load(f"data_train/chat_{split}.jsonl"), load(f"data_train/chat_lf_{split}.jsonl")
        # 1) E19's bind5 rebuilt from the same inputs must match (train hash recorded in EXPLORE_PLAN)
        b5 = old4 + chat
        if split == "train":
            random.Random(7).shuffle(b5)
        b5_now = load(f"data_train/bind5_{split}.jsonl")
        assert b5 == b5_now, f"bind5_{split} does not rebuild from bind4 + chat"
        if split == "train":
            assert sha(f"data_train/bind5_{split}.jsonl") == BIND5_TRAIN
        # 2) bind4 with list-first thoughts; everything but the answer of reason / absent think rows must be identical
        new4 = bind4_lf(split, n, seed, genres, wiki)
        rows, changed = [], collections.Counter()
        for r in old4:
            if r["kind"] in ("e12", "general"):
                rows.append(r)
                continue
            x = new4[r["id"]]
            assert {k: v for k, v in x.items() if k != "answer"} == {k: v for k, v in r.items() if k != "answer"}, r["id"]
            if x["answer"] != r["answer"]:
                assert x["kind"] in ("reason", "absent") and x["think"], r["id"]
                changed[x["kind"]] += 1
            else:
                assert not (x["kind"] in ("reason", "absent") and x["think"]), r["id"]
            rows.append(x)
        # 3) same order as bind5: shuffle the same-length list with the same seed
        out = rows + chat_lf
        if split == "train":
            random.Random(7).shuffle(out)
        assert [r["id"] for r in out] == [r["id"] for r in b5_now]
        path = f"data_train/bind6_{split}.jsonl"
        with open(path, "w", encoding="utf-8") as f:
            for r in out:
                f.write(json.dumps(r) + "\n")
        ch = sum(a["answer"] != b["answer"] for a, b in zip(out, b5_now))
        short = sum(r.get("think") and len(r["answer"].split("</think>")[0].split()) < 40
                    and "from oldest to newest" not in r["answer"] for r in out)
        print(path, len(out), sha(path), "| answers changed vs bind5:", ch, "(bind4", dict(changed), "+ chat",
              sum(a["answer"] != b["answer"] for a, b in zip(chat, chat_lf)), ") | think rows with a one-line thought, no list:", short)


if __name__ == "__main__":
    main()
