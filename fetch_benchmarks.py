"""Download the held-out benchmarks listed in PREREG_TRAINING.md (public Hugging Face datasets) into the HF cache.
Prints file sizes only."""
import os

from huggingface_hub import hf_hub_download

FILES = [  # (repo, filename, revision)
    ("ai-hyz/MemoryAgentBench", "data/Conflict_Resolution-00000-of-00001.parquet", None),
    ("baharef/ToT", "tot_semantic/test/0000.parquet", "refs/convert/parquet"),
    ("tonytan48/TempReason", "test_l2.json", None),
    ("RMT-team/babilong-1k-samples", "2k/qa1/0000.parquet", "refs/convert/parquet"),
    ("RMT-team/babilong-1k-samples", "2k/qa2/0000.parquet", "refs/convert/parquet"),
    ("RMT-team/babilong-1k-samples", "2k/qa3/0000.parquet", "refs/convert/parquet"),
    ("openai/gsm8k", "main/test/0000.parquet", "refs/convert/parquet"),
    ("Salesforce/wikitext", "wikitext-2-raw-v1/train/0000.parquet", "refs/convert/parquet"),
]

if __name__ == "__main__":
    for repo, fn, rev in FILES:
        p = hf_hub_download(repo, fn, repo_type="dataset", revision=rev)
        print(f"{repo:32s} {fn:52s} {os.path.getsize(p) / 2**20:7.1f} MB", flush=True)
