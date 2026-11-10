#!/bin/bash
# MEME prompt baselines (PREREG_MEME_PROMPTS.md), run from Git Bash: bash run_meme_prompts.sh oracle|rag
# Each stage is resumable; rerun after an interruption.
export HF_HOME='D:\hf_cache' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd "$(dirname "$0")/.."
py=.venv/Scripts/python.exe; rd="$(dirname "$0")/meme_reader.py"; log=logs/meme_prompts.log; c=$1
run() { echo "== $(date +%H:%M:%S) $c $*" >> $log; $py "$rd" answer --conds=$c "$@" >> $log 2>&1; }
if [ "$c" = oracle ]; then run; run --think; fi          # official prompt: adds the exact-recall questions
run --prompt=g11
run --prompt=g12
run --prompt=smw
run --prompt=ctrl
$py "$rd" judge >> $log 2>&1
run --prompt=g11 --think
run --prompt=g12 --think
$py "$rd" judge >> $log 2>&1
echo "ALL_DONE $c $(date +%H:%M:%S)" >> $log
