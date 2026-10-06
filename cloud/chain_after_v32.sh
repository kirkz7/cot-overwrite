#!/usr/bin/env bash
# Order (user 10-06): 32B (jobs_v32, both cards) -> per card: GPU0 14B 4-bit ConvoMem (p1b_g0) then 8B bf16 (v8),
# GPU1 14B 4-bit MemConflict/E15 (p1b_g1) then 14B bf16 (v14) -> judge everything.
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 30; done; }
run_q() { cloud/resume_queue.sh "$1"; sleep 5; wait_q "$1"; }
pgrep -f "[q]ueue_cloud.sh cloud/jobs_v32.txt" > /dev/null || run_q v32
wait_q v32
( run_q p1b_g0; until grep -q "^model Qwen/Qwen3-8B" logs/fetch_8b.log 2>/dev/null; do sleep 30; done; run_q v8 ) &
( run_q p1b_g1; run_q v14 ) &
wait
run_q judge
echo "[$(date +%H:%M:%S)] chain_after_v32 done" >> logs/queue_cloud.log
