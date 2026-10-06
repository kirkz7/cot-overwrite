"""Queue wrapper (CLOUD_PLAN P7 step 2): run gen_selfdistill.py --n 600 --engine vllm in the vLLM venv on this job's card.
Skips if data_train/selfdistill_train.jsonl already exists (the generator is not resumable, but it is one batch)."""
import os
import subprocess
import sys

os.chdir(os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
if os.path.exists("data_train/selfdistill_train.jsonl"):
    print("selfdistill_train.jsonl exists, skipping")
    sys.exit(0)
venv = os.environ.get("VLLM_VENV", "/data/vllm-venv")
env = dict(os.environ, PATH=f"{venv}/bin:/usr/local/cuda-13.0/bin:" + os.environ["PATH"], CUDA_HOME="/usr/local/cuda-13.0")
sys.exit(subprocess.call([f"{venv}/bin/python", "gen_selfdistill.py", "--n", "600", "--engine", "vllm"], env=env))
