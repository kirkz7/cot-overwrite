#!/usr/bin/env bash
# Replaces chain_after_v32.sh / after_v8.sh / rerun_v32_exteval.sh (10-07 00:10, P6 added by the user):
#   GPU0: 8B bf16 (v8, running) -> P6 base 4B -> 4B bf16 gap filler (v4)
#   GPU1: 14B 4-bit (p1b_g1, running) -> P6 E18 -> 14B bf16 (v14)
#   then: judge everything -> rerun the unfinished 32B exteval (both cards)
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 30; done; }
run_q() { cloud/resume_queue.sh "$1"; sleep 5; wait_q "$1"; }
( wait_q v8; run_q p6_base; run_q v4 ) &
( wait_q p1b_g1; run_q p6_e18; run_q v14 ) &
wait
run_q judge
run_q v32
echo "[$(date +%H:%M:%S)] orchestrate done" >> logs/queue_cloud.log
