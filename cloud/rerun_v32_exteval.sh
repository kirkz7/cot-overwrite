#!/usr/bin/env bash
# v32_exteval stopped at ToT item 277 (prompt of 42k tokens > the old max_model_len 40960; fixed in vllm_ctl.sh).
# Rerun it (the queue skips every finished v32 job; the script skips saved items) once both cards are free:
# after chain_after_v32.sh and after_v8.sh have both finished.
cd "$(dirname "$0")/.."
until grep -q "chain_after_v32 done" logs/queue_cloud.log && grep -q "after_v8 done" logs/queue_cloud.log; do sleep 60; done
cloud/resume_queue.sh v32; sleep 5
while pgrep -f "[q]ueue_cloud.sh cloud/jobs_v32.txt" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] rerun_v32_exteval done" >> logs/queue_cloud.log
