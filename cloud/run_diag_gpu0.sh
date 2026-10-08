#!/usr/bin/env bash
# pause jobs_q8a (GPU0; resumable), run the presence_penalty diagnostic there, then resume jobs_q8a
cd "$(dirname "$0")/.."
wait_q() { while pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; do sleep 15; done; }
p=$(pgrep -f "[q]ueue_cloud.sh cloud/jobs_q8a.txt" | head -1); [ -n "$p" ] && pkill -TERM -s "$(ps -o sid= -p "$p" | tr -d ' ')"
sleep 10; echo "[$(date +%H:%M:%S)] paused q8a for the presence_penalty diagnostic" >> logs/queue_cloud.log
cloud/resume_queue.sh diag; sleep 5; wait_q diag
cloud/resume_queue.sh q8a
