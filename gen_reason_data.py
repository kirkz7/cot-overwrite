"""Exploration E12 (EXPLORE_PLAN.md): LoRA data whose target is a short reasoning organized by TIME, then the answer.

Inputs are exactly the v2 training items (gen_train_data.py, same seeds; checked against data_train/*.jsonl), so E12
differs from the earlier answer-only LoRA only in the target. Target (format-agnostic wording):
    Records about <attribute>, from oldest to newest:
    - <date>: <value>            (undated items: "- <value>", ordered by the stated order)
    ...
    The newest record says <v>. | The oldest record says <v>. | The question asks about <when>; the last record before then says <v>.
    Answer: <v>
decoupled: inputs in chrono / reverse / shuffled order (v2 decoupled); control: the same inputs always chronological
(v2 chrono), with the identical target. "final" holds the bare answer (validation scores the text after "Answer:").
usage: python gen_reason_data.py
"""
import hashlib
import json
import random

import gen_train_data as g


def target(r):
    lines = [f"Records about {r['attr_name']}, from oldest to newest" + (":" if r["dated"] else " (no dates; using the stated order):")]
    for c in r["chain"]:
        lines.append(f"- {c['date']}: {c['value']}" if c["date"] else f"- {c['value']}")
    if r["qtype"] == "current":
        lines.append(f"The newest record says {r['answer']}.")
    elif r["qtype"] == "first":
        lines.append(f"The oldest record says {r['answer']}.")
    else:
        lines.append(f"The question asks about {r['query_when']}; the last record before then says {r['answer']}.")
    lines.append(f"Answer: {r['answer']}")
    return "\n".join(lines)


def main():
    wiki = g.load_wiki()
    for split, n in (("train", 5000), ("val", 600)):
        for mode in ("decoupled", "chrono"):
            rng = random.Random({"train": 11, "val": 22}[split])
            rows = [g.make_sample(rng, wiki, mode, i, g.TRAIN_FORMATS) for i in range(n)]
            old = [json.loads(l) for l in open(f"data_train/{mode}_{split}.jsonl", encoding="utf-8")]
            same = len(old) == len(rows) and all(a["prompt"] == b["prompt"] and a["answer"] == b["answer"] for a, b in zip(old, rows))
            print(mode, split, "inputs identical to the v2 file:", same, flush=True)
            assert same
            out = f"data_train/reason_{mode}_{split}.jsonl"
            with open(out, "w", encoding="utf-8") as f:
                for r in rows:
                    r2 = {k: v for k, v in r.items() if k not in ("chain",)}
                    r2["final"], r2["answer"] = r["answer"], target(r)
                    f.write(json.dumps(r2) + "\n")
            print(out, hashlib.sha256(open(out, "rb").read()).hexdigest()[:16].upper())
    d = [json.loads(l) for l in open("data_train/reason_decoupled_train.jsonl", encoding="utf-8")]
    c = [json.loads(l) for l in open("data_train/reason_chrono_train.jsonl", encoding="utf-8")]
    print("targets identical across modes:", all(a["answer"] == b["answer"] for a, b in zip(d, c)))
    print("example target:\n" + d[3]["answer"])


if __name__ == "__main__":
    main()
