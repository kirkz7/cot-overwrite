"""Audit the entire eligible SWE-chat feedback cohort without model inference."""
import json
from collections import Counter, defaultdict

import pyarrow as pa
import pyarrow.compute as pc
from pyarrow.parquet import ParquetFile

from swechat_memory import SRC, ok_result, read_content
from swechat_pivots import key, load


def cohort():
    first_write = {}
    for state in load("memory_states"):
        if state["note"].startswith("feedback") and state["via"] != "read":
            first_write.setdefault(key(state), state)
    reads = {}
    for event in load("memory_events"):
        first = first_write.get(key(event))
        if (first and event["tool"] == "Read" and first["session"] != event["session"]
                and first["ts"] < event["ts"]
                and read_content(event["result"], json.loads(event["input"] or "{}")) is not None):
            reads[event["id"]] = event
    return reads


def main():
    reads = cohort()
    sessions = {event["session"] for event in reads.values()}
    grouped = defaultdict(list)
    cols = ["session_id", "timestamp", "turn_number", "content_block_index", "role",
            "tool_call_id", "tool_name", "file_path", "content", "prompt_pushback"]
    selected = None
    for batch in ParquetFile(SRC).iter_batches(columns=cols):
        if selected is None:
            selected = pa.array(sorted(sessions), type=batch["session_id"].type)
        for row in batch.filter(pc.is_in(batch["session_id"], value_set=selected)).to_pylist():
            grouped[row["session_id"]].append(row)

    cohort_pairs = {(key(event), event["session"]) for event in reads.values()}
    observed = set()
    coverage = defaultdict(set)
    returned_reads = Counter()
    pushback_labels = Counter()
    for session, rows in grouped.items():
        rows = sorted((r for r in rows if r["timestamp"] is not None), key=lambda r: (
            r["timestamp"], r["turn_number"] or 0, r["content_block_index"] or 0
        ))
        calls = {}
        successful_edits = []
        starts = {}
        for i, row in enumerate(rows):
            cid = row["tool_call_id"]
            if row["role"] == "tool_use" and cid:
                calls[cid] = row
            elif row["role"] == "tool_result" and cid:
                if cid in reads:
                    starts.setdefault((key(reads[cid]), session), i)
                    returned_reads[cid] += 1
                call = calls.get(cid)
                if call and call["tool_name"] in {"Write", "Edit", "MultiEdit"} and ok_result(row["content"]):
                    successful_edits.append((i, (call["file_path"] or "").replace("\\", "/")))
        for pair, start in starts.items():
            observed.add(pair)
            user = [i for i in range(start + 1, len(rows)) if rows[i]["role"] == "user"]
            if user:
                coverage["user_after"].add(pair)
            if any(rows[i]["prompt_pushback"] in {"correction", "failure_report", "rejection"} for i in user):
                coverage["pushback_after"].add(pair)
            if any(i > start and path and "/memory/" not in path.lower() for i, path in successful_edits):
                coverage["returned_nonmemory_edit"].add(pair)
            note_path = "/".join(pair[0][1:])
            if user and any(i > user[0] and path == note_path for i, path in successful_edits):
                coverage["same_note_edit_after_user"].add(pair)
        if starts:
            first_start = min(starts.values())
            pushback_labels.update(row["prompt_pushback"] for row in rows[first_start + 1:]
                                   if row["role"] == "user")

    print("cohort", len(reads), "reads;", len(cohort_pairs), "pairs;",
          len({key(r) for r in reads.values()}), "notes;", len(sessions), "sessions;",
          len({r["repo"] for r in reads.values()}), "repos")
    print("distinct reads with dated return", len(returned_reads), "pairs with dated return", len(observed),
          "extra return rows", sum(returned_reads.values()) - len(returned_reads))
    print("user pushback label frequencies", dict(pushback_labels))
    for metric in ("user_after", "pushback_after", "returned_nonmemory_edit", "same_note_edit_after_user"):
        pairs = coverage[metric]
        print(metric, len(pairs), "pairs", len({pair[0][0] for pair in pairs}), "repos")


if __name__ == "__main__":
    main()
