#!/bin/bash
# Controls for the write upper bound (PREREG_MEME_WRITE.md section E), then the rest of the write-side queue, then the controls with thinking.
# Run from Git Bash: bash run_meme_write_ctrl.sh. Every stage is resumable; rerun after an interruption.
export HF_HOME='D:\hf_cache' HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8 HF_HUB_DISABLE_SYMLINKS_WARNING=1
cd "$(dirname "$0")/.."
py=.venv/Scripts/python.exe; d="$(dirname "$0")"; log=logs/meme_write.log
st() { echo "== $(date +%H:%M:%S) $*" >> $log; $py "$d/$1" "${@:2}" >> $log 2>&1; }
st meme_write_ctrl.py answer
st meme_reader.py judge
echo "CTRL_OFF_DONE $(date +%H:%M:%S)" >> $log
bash "$d/run_meme_write.sh"
st meme_write_ctrl.py answer --think
st meme_reader.py judge
echo "CTRL_THINK_DONE $(date +%H:%M:%S)" >> $log
