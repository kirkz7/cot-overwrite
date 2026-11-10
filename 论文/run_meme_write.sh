#!/bin/bash
# Write-side upper bounds + StateMem pilot (PREREG_MEME_WRITE.md), run from Git Bash: bash run_meme_write.sh
# Waits for the read-side job (run_meme_prompts.sh rag) to finish. Every stage is resumable; rerun after an interruption.
export HF_HOME='D:\hf_cache' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd "$(dirname "$0")/.."
py=.venv/Scripts/python.exe; d="$(dirname "$0")"; log=logs/meme_write.log
until grep -qa "ALL_DONE rag" logs/meme_prompts.log; do sleep 60; done
st() { echo "== $(date +%H:%M:%S) $*" >> $log; $py "$d/$1" "${@:2}" >> $log 2>&1; }
st meme_write.py answer --conds=w4,w3,w3s
st meme_write.py answer --think --conds=w4,w3,w3s
st meme_reader.py judge
echo "STEP1_PRODUCTS_DONE $(date +%H:%M:%S)" >> $log
st meme_write.py ingest
st meme_write.py smread
st meme_write.py smread --think
st meme_reader.py judge
echo "STATEMEM_PILOT_DONE $(date +%H:%M:%S)" >> $log
st meme_write.py answer --conds=w4rag,w3rag
st meme_write.py answer --think --conds=w4rag,w3rag
st meme_reader.py judge
echo "ALL_DONE write $(date +%H:%M:%S)" >> $log
