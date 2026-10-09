#!/usr/bin/env bash
# 10-07 04:55, replaces orchestrate.sh / followup.sh (GPU0 was idle while 14B bf16 ran alone on GPU1):
#   GPU0: 8B exteval rerun (jobs_v8) -> 14B bf16 all but extmem (jobs_v14_rest)
#   GPU1: 14B bf16 extmem (jobs_v14_ext, resumes)
#   then: stop the :8001 server -> judge everything -> rerun the unfinished 32B exteval (both cards)
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 30; done; }
run_q() { cloud/resume_queue.sh "$1"; sleep 5; wait_q "$1"; }
( run_q v8; run_q v14_rest ) &
( run_q v14_ext; CUDA_VISIBLE_DEVICES=1 COT_VLLM_URL=http://127.0.0.1:8001 cloud/vllm_ctl.sh down ) &
wait
run_q judge
run_q v32
echo "[$(date +%H:%M:%S)] orchestrate2 done" >> logs/queue_cloud.log
