"""Download the conversational-memory benchmarks used in the exploration phase (user approved 2026-10-02):
ConvoMem changing_evidence/2_evidence (~165 MB), PersonaMem-v1 32k (~7 MB), BEAM 100K (~5 MB)."""
import os
os.environ["HF_HUB_OFFLINE"] = "0"
from huggingface_hub import snapshot_download
for repo, pats in [("Salesforce/ConvoMem", ["README.md", "dataset_info.json", "core_benchmark/evidence_questions/changing_evidence/2_evidence/*"]),
                   ("bowen-upenn/PersonaMem-v1", ["README.md", "questions_32k.csv", "shared_contexts_32k.jsonl"]),
                   ("Mohammadta/BEAM", ["README.md", "data/100K-*"])]:
    p = snapshot_download(repo, repo_type="dataset", allow_patterns=pats)
    print(repo, "->", p, flush=True)
