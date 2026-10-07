"""Shared helpers for the application experiments (memory retrieval, logs, agent traces)."""
import gc
import json
import os
import re

import torch

from probe import greedy, load


def load_jsonl(path):
    """Rows of a jsonl file; a truncated or corrupt line (e.g. after a power cut) is dropped."""
    rows = []
    if os.path.exists(path):
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    try:
                        rows.append(json.loads(line))
                    except json.JSONDecodeError:
                        pass
    return rows


class JsonlAppender:
    """Resumable jsonl output. Keeps the valid rows already on disk (optionally filtered by `valid`),
    rewrites them cleanly, then appends. `key(row)` identifies a unit of work; `done` holds finished keys.
    Every write is flushed; the file is fsync'ed every `sync_every` writes and on close."""

    def __init__(self, path, key, valid=None, sync_every=20):
        rows = load_jsonl(path)
        if valid is not None:
            rows = valid(rows)
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r) + "\n")
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, path)
        self.key, self.rows, self.sync_every, self.n = key, rows, sync_every, 0
        self.done = {key(r) for r in rows}
        self.f = open(path, "a", encoding="utf-8")

    def write(self, *recs):
        """Write one or more rows in a single call (rows of one unit of work stay together)."""
        self.f.write("".join(json.dumps(r) + "\n" for r in recs))
        self.f.flush()
        self.n += 1
        if self.n % self.sync_every == 0:
            os.fsync(self.f.fileno())
        for r in recs:
            self.done.add(self.key(r))
            self.rows.append(r)

    def close(self):
        self.f.flush()
        os.fsync(self.f.fileno())
        self.f.close()

MODELS = {  # name -> (hf id, 4-bit)
    "Qwen3-4B": ("Qwen/Qwen3-4B", False),
    "Qwen3-14B": ("Qwen/Qwen3-14B", True),
    "OLMo-2-13B-Instruct": ("allenai/OLMo-2-1124-13B-Instruct", True),
    "Phi-4-mini": ("microsoft/Phi-4-mini-instruct", False),
    "Qwen3-1.7B": ("Qwen/Qwen3-1.7B", False),
    "Qwen3-8B": ("Qwen/Qwen3-8B", True),
    "OLMo-2-7B-Instruct": ("allenai/OLMo-2-1124-7B-Instruct", True),
}
NUM = re.compile(r"-?\d+(?:\.\d+)?")


def tag(name):
    """File-name tag of a reader: "Qwen3-4B" -> "Qwen3-4B"; "Qwen3-4B@runs/q4-dec/final" -> "Qwen3-4B+q4-dec"."""
    base, _, adapter = name.partition("@")
    if not adapter:
        return base
    parts = [p for p in re.split(r"[\\/]", adapter) if p and p not in ("final", "ckpt", "runs", ".")]
    return base + "+" + (parts[-1] if parts else "adapter")


def load_reader(name):
    """name = a MODELS key, or "<MODELS key>@<adapter dir>" for a LoRA-tuned reader (merged when the base is bf16)."""
    base, _, adapter = name.partition("@")
    hf, four = MODELS[base]
    tok, model = load(hf, four)
    if adapter:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, adapter)
        model = model if four else model.merge_and_unload()
        model.eval()
    return tok, model


def chat_prompt(tok, user, prefix="", think=False):
    """think=True (E18): Qwen3 thinking mode, the reply starts with <think>; default: thinking off (empty think block)."""
    msgs = [{"role": "user", "content": user}]
    try:
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True, enable_thinking=think)
    except TypeError:
        head = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return head + prefix


def answer(tok, model, prompt, max_new=24):
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    stop = {tok.eos_token_id} | {i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t])}
    out = tok.decode(greedy(model, ids, max_new, stop), skip_special_tokens=True)
    return out.strip().split("\n")[0]


def out_tag(name, think=False):
    """results-file tag; thinking-mode runs (E18 --think) get their own files: "Qwen3-4B+e18-dec+think"."""
    return tag(name) + ("+think" if think else "")


def to_think(prompt):
    """a thinking-off chat prompt (ends with Qwen3's empty think block) -> the same prompt with thinking on"""
    empty = "<think>\n\n</think>\n\n"
    assert prompt.endswith(empty), "not a Qwen3 thinking-off prompt"
    return prompt[: -len(empty)]


def strip_think(text):
    """The visible reply: text after </think> (thinking mode); unchanged when there is no </think>.
    An unfinished thought (budget used up inside <think>) leaves no visible reply: returns ""."""
    if not isinstance(text, str):                                    # missing field in old rows
        return ""
    if "</think>" in text:
        return text.split("</think>", 1)[1].strip()
    return "" if text.lstrip().startswith("<think>") else text


def answer_text(full):
    """Parse rule v2 (10-05, EXPLORE_PLAN E17), for judge-scored answers: ALL text after the last "Answer:" (not only its
    first line); without "Answer:", the whole response. v1 kept only the first line, so a sentence split over lines or a
    list-then-answer output reached the judge cut short."""
    full = strip_think(full).strip()
    return " ".join((full.rsplit("Answer:", 1)[1] if "Answer:" in full else full).split())


def first_number(text):
    m = NUM.search(text.replace(",", ""))
    return float(m.group()) if m else None


_STOP_CACHE = {}


def answer_fast(tok, model, prompt, max_new=24):
    """Same as answer(), caching the newline stop-set per tokenizer."""
    key = id(tok)
    if key not in _STOP_CACHE:
        _STOP_CACHE[key] = {tok.eos_token_id} | {i for t, i in tok.get_vocab().items() if "\n" in tok.convert_tokens_to_string([t])}
    ids = tok(prompt, return_tensors="pt", add_special_tokens=False).input_ids.cuda()
    out = tok.decode(greedy(model, ids, max_new, _STOP_CACHE[key]), skip_special_tokens=True)
    return out.strip().split("\n")[0]


def free_gpu(*_):
    """Release cached GPU memory. Callers must `del` their own model reference first: deleting it here
    would only drop this function's local name, and the old model would stay resident while the next one
    loads (on Windows the overflow spills into shared system memory and everything slows down)."""
    gc.collect()
    torch.cuda.empty_cache()
