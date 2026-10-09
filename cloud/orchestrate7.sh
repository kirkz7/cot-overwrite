#!/usr/bin/env bash
# 10-07 12:15, replaces orchestrate6.sh. User: the no-self-distill ablation is the top priority, after the E18 format
# comparison finishes.
#   GPU0: wait for the E18 bind4 format diagnostic -> pause E18.1 tests part 2 (jobs_e181_g0; resumable) -> ablation
#         training + format test (jobs_nosd) -> resume jobs_e181_g0
#   GPU1: E18.1 tests part 1 (jobs_e181_g1, running; PersonaMem main gate) -> P6b E18
#   then: judge everything -> rerun the unfinished 32B exteval (both cards)
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 30; done; }
run_q() { cloud/resume_queue.sh "$1"; sleep 5; wait_q "$1"; }
stop_q() { p=$(pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" | head -1); [ -n "$p" ] && pkill -TERM -s "$(ps -o sid= -p "$p" | tr -d ' ')"; sleep 10
           echo "[$(date +%H:%M:%S)] paused $1" >> logs/queue_cloud.log; }
( while pgrep -f "[e]xplore_format_eval.py run --data bind4 --modes think,direct --models Qwen3-4B@runs/e18-dec" > /dev/null; do sleep 20; done
  stop_q e181_g0; run_q nosd; run_q e181_g0 ) &
( wait_q e181_g1; run_q p6b_e18 ) &
wait
run_q judge
run_q v32
echo "[$(date +%H:%M:%S)] orchestrate7 done" >> logs/queue_cloud.log
