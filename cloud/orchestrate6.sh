#!/usr/bin/env bash
# 10-07 08:45, replaces orchestrate5.sh (E18.1 training failed once: CheckpointError, fixed in train_lora.py).
# E18.1 tests start only after "end   e181_train exit=0"; if training fails again, everything E18.1 stops there.
#   GPU1: E18.1 training (jobs_e181; data steps already done) -> tests part 1 (jobs_e181_g1) -> P6b E18
#   GPU0: P6b base 4B (running) -> tests part 2 (jobs_e181_g0)
#   then: judge everything -> rerun the unfinished 32B exteval (both cards)
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 30; done; }
run_q() { cloud/resume_queue.sh "$1"; sleep 5; wait_q "$1"; }
trained() { grep -q "end   e181_train exit=0" logs/queue_cloud.log; }
( run_q e181; trained && { run_q e181_g1; run_q p6b_e18; } ) &
( wait_q p6b_base
  until grep -q "queue done cloud/jobs_e181.txt" <(grep -A9999 "orchestrate6 start" logs/queue_cloud.log); do sleep 60; done
  trained && run_q e181_g0 ) &
wait
trained || { echo "[$(date +%H:%M:%S)] orchestrate6: E18.1 training failed, stopped" >> logs/queue_cloud.log; exit 1; }
run_q judge
run_q v32
echo "[$(date +%H:%M:%S)] orchestrate6 done" >> logs/queue_cloud.log
