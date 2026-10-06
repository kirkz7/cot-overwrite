#!/usr/bin/env bash
# Gap filler (user 10-06 21:40): when the 8B queue on GPU0 is done, run the 4B full suite there (jobs_v4), then a second
# judging pass once the main chain (chain_after_v32.sh) has finished its own judging.
cd "$(dirname "$0")/.."
until grep -q "queue done cloud/jobs_v8.txt" logs/queue_cloud.log; do sleep 60; done
cloud/resume_queue.sh v4; sleep 5
while pgrep -f "[q]ueue_cloud.sh cloud/jobs_v4.txt" > /dev/null; do sleep 30; done
until grep -q "chain_after_v32 done" logs/queue_cloud.log; do sleep 60; done
cloud/resume_queue.sh judge2; sleep 5
while pgrep -f "[q]ueue_cloud.sh cloud/jobs_judge2.txt" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] after_v8 done" >> logs/queue_cloud.log
