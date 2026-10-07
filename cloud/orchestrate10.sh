#!/usr/bin/env bash
# 10-07 21:45, replaces orchestrate9.sh: Gemma-3-27B (running) -> judge pass 2 -> 8B E18.1 on both cards (jobs_q8a GPU0, jobs_q8b GPU1).
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 30; done; }
run_q() { cloud/resume_queue.sh "$1"; sleep 5; wait_q "$1"; }
wait_q g27
run_q judge2
COT_VLLM_URL=http://127.0.0.1:8000 cloud/vllm_ctl.sh down
( run_q q8a ) &
( run_q q8b ) &
wait
echo "[$(date +%H:%M:%S)] orchestrate10 done" >> logs/queue_cloud.log
