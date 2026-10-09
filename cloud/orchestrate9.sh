#!/usr/bin/env bash
# 10-07 17:00, replaces orchestrate8.sh: 27B failed to start (the 12B server on :8001 was still up; vllm_ctl.sh now stops
# other servers before an all-card start). Order: 32B exteval rerun (running) -> Gemma-3-27B -> judge pass 2.
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 30; done; }
run_q() { cloud/resume_queue.sh "$1"; sleep 5; wait_q "$1"; }
wait_q v32
run_q g27
run_q judge2
echo "[$(date +%H:%M:%S)] orchestrate9 done" >> logs/queue_cloud.log
