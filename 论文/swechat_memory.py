"""Reconstruct Claude Code memory-note states from SWE-chat and pair MEMORY.md index lines with the notes they link to.

Data: SALT-NLP/SWE-chat `conversations.parquet` (local copy under D:\\hf_cache\\manual\\swe-chat\\).
A memory store is one directory `~/.claude/projects/<slug>/memory/` inside one repo. Its notes are seen only through the
agent's own tool calls:
  Read    the tool-result row carries the whole file (line-numbered) unless offset/limit was passed
  Write   the tool input carries the whole file
  Edit    the tool input carries old_string/new_string; applied to the last known state when that state is known
  MultiEdit  a list of edits, applied in order
Failed calls (error results) and reads that return notices instead of content are dropped.

  python 论文/swechat_memory.py extract   -> work/memory_events.jsonl   (one row per memory-file tool call, with result)
  python 论文/swechat_memory.py states    -> work/memory_states.jsonl   (one row per known full state of a note)
  python 论文/swechat_memory.py pairs     -> work/index_pairs_full.json (distinct index-line / note-state triples)
No note text is printed by any command; counts only.
"""
import collections
import hashlib
import json
import os
import re
import sys

SRC = r"D:\hf_cache\manual\swe-chat\conversations.parquet"
WORK = r"D:\hf_cache\manual\swe-chat-work"
MEM_DIR = re.compile(r"^(.*[\\/]\.claude[\\/]projects[\\/][^\\/]+[\\/]memory)[\\/](.+)$", re.I)
TOOLS = {"Read", "Write", "Edit", "MultiEdit"}


def extract():
    import pyarrow.parquet as pq
    os.makedirs(WORK, exist_ok=True)
    f = pq.ParquetFile(SRC)
    use_cols = ["session_id", "repo_id", "turn_number", "content_block_index", "role", "tool_name", "tool_call_id",
                "file_path", "timestamp", "tool_input_json"]
    calls = {}
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=use_cols).to_pylist()
        for r in t:
            if r["role"] != "tool_use" or r["tool_name"] not in TOOLS or not r["file_path"]:
                continue
            m = MEM_DIR.match(r["file_path"].replace("\\", "/"))
            if not m:
                continue
            calls[r["tool_call_id"]] = dict(session=r["session_id"], repo=r["repo_id"], turn=r["turn_number"],
                                            block=r["content_block_index"], tool=r["tool_name"],
                                            store=m.group(1), note=m.group(2), ts=r["timestamp"].isoformat(),
                                            input=r["tool_input_json"])
    print("memory tool calls:", len(calls))
    results = {}
    for i in range(f.num_row_groups):
        t = f.read_row_group(i, columns=["role", "tool_call_id", "content"]).to_pylist()
        for r in t:
            if r["role"] == "tool_result" and r["tool_call_id"] in calls:
                results[r["tool_call_id"]] = r["content"]
    print("with a result row:", len(results))
    with open(os.path.join(WORK, "memory_events.jsonl"), "w", encoding="utf-8") as w:
        for cid, c in calls.items():
            c["id"] = cid
            c["result"] = results.get(cid)
            w.write(json.dumps(c, ensure_ascii=False) + "\n")


NUMBERED = re.compile(r"^\s*\d+(?:\u2192|\t)", re.M)
REMINDER = re.compile(r"<system-reminder>.*?</system-reminder>", re.S)


def read_content(result, inp):
    """whole-file content from a Read result, or None.
    Two line-number formats occur (`   1\u2192text` and `1<TAB>text`); the harness may wrap the content in
    <system-reminder> blocks (e.g. a note that the memory is N days old), which are removed."""
    if not result or inp.get("offset") or inp.get("limit"):
        return None
    if result.startswith(("<tool_use_error>", "File does not exist", "File unchanged", "Error")):
        return None
    body = REMINDER.sub("", result).strip("\n")
    if not NUMBERED.search(body):
        return "" if "empty" in body.lower()[:200] else None
    lines = []
    for line in body.split("\n"):
        m = re.match(r"^\s*\d+(?:\u2192|\t)(.*)$", line)
        if m:
            lines.append(m.group(1))
    return "\n".join(lines)


def reminder_ages(result):
    """days stated in a harness 'This memory is N days old' reminder, if any"""
    m = re.search(r"This memory is (\d+) days? old", result or "")
    return int(m.group(1)) if m else None


def ok_result(result):
    return result is not None and not result.startswith(("<tool_use_error>", "Error", "File has not been read"))


def states():
    ev = [json.loads(l) for l in open(os.path.join(WORK, "memory_events.jsonl"), encoding="utf-8")]
    ev.sort(key=lambda e: (e["store"], e["note"], e["ts"], e["session"], e["turn"], e["block"]))
    out, stats = [], collections.Counter()
    cur = {}
    for e in ev:
        key = (e["repo"], e["store"], e["note"])
        try:
            inp = json.loads(e["input"]) if e["input"] else {}
        except json.JSONDecodeError:
            stats["bad_input"] += 1
            continue
        new, via = None, None
        if e["tool"] == "Read":
            new, via = read_content(e["result"], inp), "read"
            if new is None:
                stats["read_unusable"] += 1
        elif e["tool"] == "Write":
            if ok_result(e["result"]) and isinstance(inp.get("content"), str):
                new, via = inp["content"], "write"
            else:
                stats["write_failed"] += 1
        else:
            edits = inp.get("edits") if e["tool"] == "MultiEdit" else [inp]
            base = cur.get(key)
            if not ok_result(e["result"]):
                stats["edit_failed"] += 1
            elif base is None:
                stats["edit_no_base"] += 1
                cur.pop(key, None)
            else:
                s = base
                for ed in edits or []:
                    old, nw = ed.get("old_string", ""), ed.get("new_string", "")
                    if old and old in s:
                        s = s.replace(old, nw) if ed.get("replace_all") else s.replace(old, nw, 1)
                    else:
                        s = None
                        break
                if s is None:
                    stats["edit_mismatch"] += 1
                    cur.pop(key, None)
                else:
                    new, via = s, "edit"
        if new is not None:
            cur[key] = new
            stats[via] += 1
            out.append(dict(repo=e["repo"], store=e["store"], note=e["note"], ts=e["ts"], session=e["session"], via=via,
                            sha=hashlib.sha1(new.encode("utf-8")).hexdigest()[:12], text=new))
    with open(os.path.join(WORK, "memory_states.jsonl"), "w", encoding="utf-8") as w:
        for o in out:
            w.write(json.dumps(o, ensure_ascii=False) + "\n")
    notes = {(o["repo"], o["store"], o["note"]) for o in out}
    print("events", len(ev), "| known states", len(out), "| notes", len(notes),
          "| stores", len({(o["repo"], o["store"]) for o in out}), "| repos", len({o["repo"] for o in out}))
    print("by source:", dict(stats))


LINK = re.compile(r"^\s*[-*]\s*\[([^\]]+)\]\(([^)]+\.md)\)\s*[\u2014\u2013:-]+\s*(.+)$")


def desc_of(t):
    m = re.search(r"^description:\s*(.+)$", t, re.M)
    return m.group(1).strip() if m else None


def pairs():
    st = [json.loads(l) for l in open(os.path.join(WORK, "memory_states.jsonl"), encoding="utf-8")]
    by_store = collections.defaultdict(list)
    for s in st:
        by_store[s["repo"], s["store"]].append(s)
    seen, items, n_lines, n_linked = set(), [], 0, 0
    for (repo, store), obs in sorted(by_store.items()):
        obs.sort(key=lambda s: s["ts"])
        notes = collections.defaultdict(list)
        for s in obs:
            notes[s["note"]].append(s)
        for s in obs:
            if os.path.basename(s["note"]).lower() != "memory.md":
                continue
            for line in s["text"].splitlines():
                m = LINK.match(line)
                if not m:
                    continue
                n_lines += 1
                target = m.group(2).split("/")[-1]
                prev = [o for o in notes.get(target, []) if o["ts"] <= s["ts"]]
                if not prev:
                    continue
                n_linked += 1
                note = prev[-1]
                key = (repo, store, target, m.group(3).strip(), desc_of(note["text"]))
                if key in seen:
                    continue
                seen.add(key)
                items.append(dict(repo=repo, store=store, note=target, idx_ts=s["ts"], note_ts=note["ts"], idx_via=s["via"],
                                  note_via=note["via"], idx=m.group(3).strip(), ndesc=desc_of(note["text"]),
                                  ntext=note["text"][:4000]))
    json.dump(items, open(os.path.join(WORK, "index_pairs_full.json"), "w", encoding="utf-8"), ensure_ascii=False)
    kind = collections.Counter(i["note"].split("_")[0] if "_" in i["note"] else "other" for i in items)
    print("index link lines", n_lines, "| with the note known at or before", n_linked, "| distinct triples", len(items))
    print("by note type:", kind.most_common(8))
    print("stores", len({(i['repo'], i['store']) for i in items}), "| repos", len({i['repo'] for i in items}),
          "| triples in the largest store", max(collections.Counter((i['repo'], i['store']) for i in items).values()))


if __name__ == "__main__":
    {"extract": extract, "states": states, "pairs": pairs}[sys.argv[1]]()
