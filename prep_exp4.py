"""Build the real-trace set for Exp 4 from OpenR1-Math-220k (default shard 0).

Keeps DeepSeek-R1 traces with an integer answer that math-verify marked correct,
strips the conclusion (paragraphs with \\boxed / "final answer"), and tags each trace:
  rev   = an earlier conclusion-like statement gives an integer != gold, and the trace has
          retraction markers  (numerical self-revision)
  clean = every conclusion-like statement gives gold
"""
import json
import re

import pandas as pd
from huggingface_hub import hf_hub_download

MAX_CHARS = 16_000
# answer candidates: "answer ... is X" or \boxed{X}, searched in the unstripped trace
CONCL = re.compile(r"(?i)(?:\banswer\b[^.\n]{0,30}?(?:\bis\b|=|\bbe\b)\s*\$?(?:\\boxed\{)?|\\boxed\{)\s*(-?\d+)(?!\d|\.\d)")
RETRACT = re.compile(r"(?i)\b(?:wait|mistake|wrong|incorrect|miscalculat\w*|actually)\b")
INT = re.compile(r"(?<![\d.])-?\d+(?!\d|\.\d)")


def strip_conclusion(think):
    paras = [p.strip() for p in think.split("\n\n") if p.strip()]
    return [p for p in paras if "\\boxed" not in p and "final answer" not in p.lower()]


def main():
    df = pd.concat([pd.read_parquet(hf_hub_download("open-r1/OpenR1-Math-220k", f"data/train-{i:05d}-of-00010.parquet", repo_type="dataset")) for i in range(5)])
    df = df[df.answer.str.fullmatch(r"-?\d+") & (df.answer.str.len() <= 7)]
    out = []
    for _, r in df.iterrows():
        gens = list(r.generations)
        fins = list(r.finish_reasons) if r.finish_reasons is not None else [None] * len(gens)
        for g, ok, fin in zip(gens, r.correctness_math_verify, fins):
            if not ok or "</think>" not in g or fin not in (None, "stop"):
                continue
            think = g.split("<think>", 1)[-1].split("</think>", 1)[0]
            if len(think) > MAX_CHARS:
                continue
            paras = strip_conclusion(think)
            text = "\n\n".join(paras)
            gold = int(r.answer)
            if gold not in {int(x) for x in INT.findall(text)} or len(paras) < 6:
                continue
            concl = [int(x) for x in CONCL.findall(think)]
            # stale candidates must survive stripping, otherwise the probe can't pick them
            present = {int(x) for x in INT.findall(text)}
            if abs(gold) < 10:  # small ints appear everywhere; not diagnostic
                continue
            stale = sorted({c for c in concl if c != gold and c in present and abs(c) >= 10})
            if stale and RETRACT.search(text):
                group = "rev"
            elif concl and not stale:
                group = "clean"
            elif not concl:
                group = "unk"
            else:
                continue
            out.append(dict(uuid=r.uuid, problem=r.problem, gold=gold, group=group, stale=stale,
                            paras=paras, n_paras=len(paras), chars=len(text)))
            break  # one trace per problem
    with open("data_exp4.jsonl", "w", encoding="utf-8") as f:
        for o in out:
            f.write(json.dumps(o, ensure_ascii=False) + "\n")
    d = pd.DataFrame(out)
    print(d.group.value_counts())
    print(d.groupby("group")[["n_paras", "chars"]].median())


if __name__ == "__main__":
    main()

