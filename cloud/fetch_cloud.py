"""Download, on the cloud machine, the model / dataset files CLOUD_PLAN.md section 3 lists, into $HF_HOME/hub and
$COT_DATA. Revisions the desktop pinned (laptop_fetch.py on the laptop branch) are reused; the rest take the current
main and the resolved commit is printed, so it can be recorded in CLOUD_NOTEBOOK.md and checked against the desktop.
usage: .venv/bin/python cloud/fetch_cloud.py [--olmo]      (run with HF_HOME / COT_DATA set as in cloud/queue_cloud.sh)
"""
import argparse
import hashlib
import os
import sys
import urllib.request

os.environ.pop("HF_HUB_OFFLINE", None)
os.environ.pop("HF_DATASETS_OFFLINE", None)
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from huggingface_hub import HfApi, hf_hub_download, snapshot_download

from paths import data, hub

MODEL_FILES = ["*.json", "*.safetensors", "*.txt", "*.model", "*.py", "*.tiktoken"]   # as download_models.py
MODELS = ["Qwen/Qwen3-4B", "Qwen/Qwen3-14B", "Qwen/Qwen3-32B"]
OLMO = "allenai/OLMo-2-0325-32B-Instruct"
# (repo, revision used on the desktop or None = current main, files, ref the scripts ask for)
DATASETS = [
    ("xiaowu0162/longmemeval-cleaned", "98d7416c24c778c2fee6e6f3006e7a073259d48f", ["longmemeval_oracle.json"], "main"),
    ("ai-hyz/MemoryAgentBench", "7ea066982b140a19337e17e60d45d4076e042faf",
     ["data/Conflict_Resolution-00000-of-00001.parquet"], "main"),
    ("baharef/ToT", "9aed387ad0eaf1a1aa321862ae8a02ffc12076d0", ["tot_semantic/test/0000.parquet"], "refs/convert/parquet"),
    ("tonytan48/TempReason", "1646dea364ac667dd6098646da7d6638de00cc71", ["test_l2.json"], "main"),
    ("RMT-team/babilong-1k-samples", "d988c9412bad836bb09ede913a1b367046ebdb78",
     ["2k/qa1/0000.parquet", "2k/qa2/0000.parquet", "2k/qa3/0000.parquet"], "refs/convert/parquet"),
    ("openai/gsm8k", "a05f38c23a0e9ab0b71de8a2b4947e20f74f68f7", ["main/test/0000.parquet"], "refs/convert/parquet"),
    ("Salesforce/wikitext", None, ["wikitext-2-raw-v1/train/0000.parquet"], "refs/convert/parquet"),
]
SNAPSHOTS = [   # as fetch_convo.py
    ("Salesforce/ConvoMem", ["README.md", "dataset_info.json", "core_benchmark/evidence_questions/changing_evidence/2_evidence/*"]),
    ("bowen-upenn/PersonaMem-v1", ["README.md", "questions_32k.csv", "shared_contexts_32k.jsonl"]),
]
GITHUB = [   # (url, local path under COT_DATA)
    ("https://raw.githubusercontent.com/TaoZhen1110/MemConflict/main/Data/Step4_4.jsonl", ("memconflict", "Step4_4.jsonl")),
    ("https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json", ("locomo", "locomo10.json")),
]


def write_ref(repo, rtype, ref, rev):
    """offline runs ask for `ref`; point it at the pinned commit, exactly as on the desktop"""
    p = hub(f"{rtype}s--{repo.replace('/', '--')}", "refs", *ref.split("/"))
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as fh:
        fh.write(rev)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--olmo", action="store_true", help="also OLMo-2-32B-Instruct (P3, only after the user agrees)")
    args = ap.parse_args()
    api = HfApi()
    for repo, rev, files, ref in DATASETS:
        rev = rev or api.dataset_info(repo, revision=ref).sha
        for f in files:
            hf_hub_download(repo, f, repo_type="dataset", revision=rev)
        write_ref(repo, "dataset", ref, rev)
        print("dataset", repo, ref, "->", rev, flush=True)
    for repo, pats in SNAPSHOTS:
        p = snapshot_download(repo, repo_type="dataset", allow_patterns=pats)
        print("dataset", repo, "main ->", os.path.basename(p), flush=True)
    for url, parts in GITHUB:
        dst = data(*parts)
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        if not os.path.exists(dst):
            urllib.request.urlretrieve(url, dst + ".tmp")
            os.replace(dst + ".tmp", dst)
        print("file", dst, "sha256", hashlib.sha256(open(dst, "rb").read()).hexdigest(), flush=True)
    for repo in MODELS + ([OLMO] if args.olmo else []):
        p = snapshot_download(repo, allow_patterns=MODEL_FILES)
        print("model", repo, "main ->", os.path.basename(p), flush=True)
    print("all files in", hub(), "and", data())


if __name__ == "__main__":
    main()
