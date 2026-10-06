#!/usr/bin/env bash
# 10-06 order (user): 32B (jobs_v32) -> 14B 4-bit P1b on both cards (jobs_p1b_g0 + jobs_p1b_g1) -> judge everything.
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 30; done; }
cloud/resume_queue.sh v32; sleep 5; wait_q v32
cloud/resume_queue.sh p1b_g0; cloud/resume_queue.sh p1b_g1; sleep 5; wait_q p1b_g0; wait_q p1b_g1
cloud/resume_queue.sh judge; sleep 5; wait_q judge
echo "[$(date +%H:%M:%S)] chain_after_v32 done" >> logs/queue_cloud.log
