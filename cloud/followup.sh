#!/usr/bin/env bash
# After orchestrate.sh: rerun the 8B exteval that stopped on the 42k-token ToT prompts (now skipped under vLLM).
cd "$(dirname "$0")/.."
until grep -q "orchestrate done" logs/queue_cloud.log; do sleep 60; done
cloud/resume_queue.sh v8; sleep 5
while pgrep -f "[q]ueue_cloud.sh cloud/jobs_v8.txt" > /dev/null; do sleep 30; done
echo "[$(date +%H:%M:%S)] followup done" >> logs/queue_cloud.log
