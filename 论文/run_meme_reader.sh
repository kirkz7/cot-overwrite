#!/bin/bash
# MEME reader pilot (PREREG_MEME_READER.md), run from Git Bash. Each stage is resumable; rerun after an interruption.
export HF_HOME='D:\hf_cache' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd "$(dirname "$0")/.."
py=.venv/Scripts/python.exe; rd="$(dirname "$0")/meme_reader.py"; log=logs/meme_reader.log
$py "$rd" answer                         >> $log 2>&1
$py "$rd" judge                          >> $log 2>&1
$py "$rd" answer --think --conds=oracle  >> $log 2>&1
$py "$rd" judge                          >> $log 2>&1
$py "$rd" answer --think --conds=rag     >> $log 2>&1
$py "$rd" judge                          >> $log 2>&1
echo ALL_DONE >> $log
