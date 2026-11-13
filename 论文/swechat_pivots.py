"""Local SWE-chat feasibility checks. Emit aggregate counts only."""
import json
import re
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

from swechat_memory import WORK, read_content, reminder_ages


def load(name):
    with (Path(WORK) / f"{name}.jsonl").open(encoding="utf-8") as stream:
        return sorted(map(json.loads, stream), key=lambda row: row["ts"])


def key(row):
    return row["repo"], row["store"], row["note"]


def words(row):
    return set(re.findall(r"\w+", row["text"].lower()))


def main():
    states = load("memory_states")
    events = load("memory_events")
    first_write, prior, edits = {}, {}, defaultdict(list)
    reads = defaultdict(list)
    for row in states:
        note = key(row)
        if row["via"] == "read":
            reads[note].append(row)
        else:
            first_write.setdefault(note, row)
            old = prior.get(note)
            if old and old["sha"] != row["sha"] and note[2].lower() != "memory.md":
                edits[(*note[:2], row["session"])].append(row)
        prior[note] = row

    feedback = {note: row for note, row in first_write.items() if note[2].startswith("feedback")}
    eligible = {note: [read for read in reads[note] if read["ts"] > row["ts"]
                       and read["session"] != row["session"]] for note, row in feedback.items()}
    eligible = {note: rows for note, rows in eligible.items() if rows}
    print("A", len(feedback), len(eligible), len({note[0] for note in eligible}),
          len({row["session"] for rows in eligible.values() for row in rows}))
    from pyarrow.parquet import ParquetFile
    from swechat_memory import SRC
    first_read = {}
    for rows in eligible.values():
        for row in rows:
            first_read.setdefault(row["session"], row["ts"])
            first_read[row["session"]] = min(first_read[row["session"]], row["ts"])
    follow_edits = set()
    for batch in ParquetFile(SRC).iter_batches(columns=["session_id", "tool_name", "file_path", "timestamp", "role"]):
        for row in batch.to_pylist():
            stamp = first_read.get(row["session_id"])
            if (stamp and row["role"] == "tool_use" and row["tool_name"] in {"Edit", "MultiEdit", "Write"}
                    and row["file_path"] and "/memory/" not in row["file_path"].replace("\\", "/").lower()
                    and row["timestamp"].isoformat() > stamp):
                follow_edits.add(row["session_id"])
    print("A follow-up edit sessions", len(follow_edits))

    by_store = defaultdict(list)
    for row in states:
        by_store[key(row)[:2]].append(row)
    scores = Counter()
    for group, changes in edits.items():
        source = changes[0]
        target = next((row for row in changes[1:] if key(row) != key(source)), None)
        if target is None:
            continue
        pool = {}
        for row in by_store[group[:2]]:
            if row["ts"] >= source["ts"]:
                break
            if row["note"].lower() != "memory.md" and row["note"] != source["note"]:
                pool[row["note"]] = row
        if target["note"] not in pool or len(pool) <= 3:
            continue
        tokens = words(source)
        recent = sorted(pool, key=lambda note: (pool[note]["ts"], note), reverse=True)
        def similarity(note):
            other = words(pool[note])
            return len(tokens & other) / max(1, len(tokens | other))
        lexical = sorted(recent, key=similarity, reverse=True)
        scores["n"] += 1
        scores["recent_top3"] += target["note"] in recent[:3]
        scores["lexical_top3"] += target["note"] in lexical[:3]
    print("B", dict(scores))

    timeline = defaultdict(list)
    writes = {}
    by_age = defaultdict(Counter)
    for row in events:
        note = key(row)
        stamp = datetime.fromisoformat(row["ts"])
        if row["tool"] != "Read":
            continue
        if read_content(row["result"], json.loads(row["input"] or "{}")) is not None:
            timeline[(*note[:2], row["session"])].append((stamp, row))
    for row in states:
        if row["via"] != "read":
            writes.setdefault(key(row), []).append(datetime.fromisoformat(row["ts"]))
    from bisect import bisect_left
    for sequence in timeline.values():
        for index, (stamp, row) in enumerate(sequence):
            age = reminder_ages(row["result"])
            history = writes.get(key(row), [])
            previous = bisect_left(history, stamp) - 1
            if previous < 0:
                continue
            inferred = (stamp - history[previous]).days
            if age is not None and age != inferred:
                continue
            follow = any(other["note"] != row["note"] for later, other in sequence[index + 1:]
                         if (later - stamp).total_seconds() <= 300)
            by_age[inferred][(age is not None, follow)] += 1
    totals = Counter()
    matched = Counter()
    weighted_difference = 0.0
    total_weight = 0
    for counts in by_age.values():
        totals.update(counts)
        control = counts[False, False] + counts[False, True]
        reminder = counts[True, False] + counts[True, True]
        if min(control, reminder) >= 5:
            matched.update(counts)
            weight = control + reminder
            total_weight += weight
            weighted_difference += weight * (counts[True, True] / reminder - counts[False, True] / control)
    print("C", dict(totals), "overlap", dict(matched))
    if min(sum(matched[arm, outcome] for outcome in (False, True)) for arm in (False, True)) >= 30:
        print("C age-standardized reminder minus control, pp", 100 * weighted_difference / total_weight)


if __name__ == "__main__":
    main()
