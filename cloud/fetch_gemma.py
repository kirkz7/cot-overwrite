"""Second family (user 10-07): Gemma 3 instruction-tuned 12B and 27B (gated; the user accepted the license and logged in)."""
import os
os.environ.pop("HF_HUB_OFFLINE", None)
from huggingface_hub import snapshot_download
for repo, rev in (("google/gemma-3-12b-it", "96b6f1eccf38110c56df3a15bffe176da04bfd80"[:0] or None),
                  ("google/gemma-3-27b-it", None)):
    p = snapshot_download(repo, revision=rev, allow_patterns=["*.json", "*.safetensors", "*.model", "*.txt", "*.jinja"])
    print("model", repo, "->", os.path.basename(p), flush=True)
