"""Download, on the laptop, exactly the public model / dataset files the desktop uses, pinned to the desktop's revisions,
into D:\\hf_cache\\hub (run_app_memory.py reads LongMemEval from that fixed path). No HuggingFace login needed.
usage: .venv\\Scripts\\python.exe laptop_fetch.py
"""
import os

os.environ.pop("HF_HUB_OFFLINE", None)
from huggingface_hub import hf_hub_download

CACHE = r"D:\hf_cache\hub"
# (repo, type, revision used on the desktop, files cached on the desktop)
PINS = [
    ("Qwen/Qwen3-1.7B", "model", "70d244cc86ccca08cf5af4e1e306ecf908b1ad5e",
     ["config.json", "generation_config.json", "merges.txt", "model-00001-of-00002.safetensors",
      "model-00002-of-00002.safetensors", "model.safetensors.index.json", "tokenizer.json", "tokenizer_config.json",
      "vocab.json"]),
    ("xiaowu0162/longmemeval-cleaned", "dataset", "98d7416c24c778c2fee6e6f3006e7a073259d48f", ["longmemeval_oracle.json"]),
    ("ai-hyz/MemoryAgentBench", "dataset", "7ea066982b140a19337e17e60d45d4076e042faf",
     ["data/Conflict_Resolution-00000-of-00001.parquet"]),
    ("baharef/ToT", "dataset", "9aed387ad0eaf1a1aa321862ae8a02ffc12076d0", ["tot_semantic/test/0000.parquet"]),
    ("tonytan48/TempReason", "dataset", "1646dea364ac667dd6098646da7d6638de00cc71", ["test_l2.json"]),
    ("RMT-team/babilong-1k-samples", "dataset", "d988c9412bad836bb09ede913a1b367046ebdb78",
     ["2k/qa1/0000.parquet", "2k/qa2/0000.parquet", "2k/qa3/0000.parquet"]),
    ("openai/gsm8k", "dataset", "a05f38c23a0e9ab0b71de8a2b4947e20f74f68f7", ["main/test/0000.parquet"]),
]

# the evaluation scripts ask for these refs (branch names) and the laptop queue runs offline (HF_HUB_OFFLINE=1),
# so the cache must map each ref to the pinned commit, exactly as on the desktop
REFS = {"baharef/ToT": "refs/convert/parquet", "RMT-team/babilong-1k-samples": "refs/convert/parquet",
        "openai/gsm8k": "refs/convert/parquet"}   # all others: main

for repo, rtype, rev, files in PINS:
    for f in files:
        p = hf_hub_download(repo, f, repo_type=rtype, revision=rev, cache_dir=CACHE)
        print("ok", repo, f, os.path.getsize(p) // 1024, "KB")
    ref = os.path.join(CACHE, f"{rtype}s--{repo.replace('/', '--')}", "refs", *REFS.get(repo, "main").split("/"))
    os.makedirs(os.path.dirname(ref), exist_ok=True)
    with open(ref, "w") as fh:
        fh.write(rev)
    print("ref", repo, REFS.get(repo, "main"), "->", rev[:12])
print("all files in", CACHE)
