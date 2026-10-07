#!/usr/bin/env bash
# 10-07 13:40, replaces orchestrate7.sh. User: second family (Gemma 3) has priority; cloud only for large models.
#   GPU0: E18.1 MemConflict gate (jobs_e181_g0, running) -> [12B downloaded] -> Gemma-3-12B part A (jobs_g12a)
#   GPU1: E18.1 PersonaMem gate (jobs_e181_g1, running) -> judge pass 1 on GPU1 (jobs_judge) -> Gemma-3-12B part B (jobs_g12b)
#   then both cards: [27B downloaded] -> Gemma-3-27B (jobs_g27) -> 32B exteval rerun (jobs_v32) -> judge pass 2 (jobs_judge2)
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 30; done; }
run_q() { cloud/resume_queue.sh "$1"; sleep 5; wait_q "$1"; }
have() { until grep -q "^model $1 " logs/fetch_gemma.log; do sleep 60; done; }
( wait_q e181_g0; have google/gemma-3-12b-it; run_q g12a ) &
( wait_q e181_g1; wait_q e181_g0; run_q judge; have google/gemma-3-12b-it; run_q g12b ) &
wait
have google/gemma-3-27b-it
run_q g27
run_q v32
run_q judge2
echo "[$(date +%H:%M:%S)] orchestrate8 done" >> logs/queue_cloud.log
