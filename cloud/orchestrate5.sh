#!/usr/bin/env bash
# 10-07 07:00, replaces orchestrate4.sh (user: balance the two cards):
#   GPU1: 14B bf16 extmem + exteval (jobs_v14_ext, running) -> stop :8001 -> E18.1 data + training (jobs_e181)
#         -> E18.1 tests part 1 (jobs_e181_g1) -> P6b E18 (lowest priority)
#   GPU0: 14B bf16 rest (jobs_v14_rest, running) -> P6b base 4B -> [wait for the E18.1 training] -> E18.1 tests part 2 (jobs_e181_g0)
#   then: judge everything -> rerun the unfinished 32B exteval (both cards)
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 30; done; }
run_q() { cloud/resume_queue.sh "$1"; sleep 5; wait_q "$1"; }
( wait_q v14_rest; run_q p6b_base
  until grep -q "queue done cloud/jobs_e181.txt" logs/queue_cloud.log; do sleep 60; done
  run_q e181_g0 ) &
( wait_q v14_ext; CUDA_VISIBLE_DEVICES=1 COT_VLLM_URL=http://127.0.0.1:8001 cloud/vllm_ctl.sh down
  run_q e181; run_q e181_g1; run_q p6b_e18 ) &
wait
run_q judge
run_q v32
echo "[$(date +%H:%M:%S)] orchestrate5 done" >> logs/queue_cloud.log
