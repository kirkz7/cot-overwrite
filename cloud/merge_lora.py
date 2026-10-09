"""Merge a LoRA into its bf16 base and save a plain model dir, so vLLM serves it like any model (cloud 10-07, 8B E18.1).
Same merge as app_common.load_reader (PeftModel ... merge_and_unload).
usage: python cloud/merge_lora.py Qwen3-8B-bf16 runs/e181-q8/final runs/e181-q8/merged"""
import os, sys
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
from app_common import load_reader
name, adapter, out = sys.argv[1:4]
if os.path.exists(os.path.join(out, "config.json")):
    print("merged model exists:", out); sys.exit(0)
tok, model = load_reader(f"{name}@{adapter}")
model.save_pretrained(out, safe_serialization=True)
tok.save_pretrained(out)
print("saved", out)
