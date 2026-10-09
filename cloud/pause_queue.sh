#!/usr/bin/env bash
# Stop the running cloud queue and its current job. Saved items are kept; cloud/resume_queue.sh continues.
cd "$(dirname "$0")/.."
pid=$(pgrep -f "cloud/queue_cloud.sh" | head -1)
[ -z "$pid" ] && { echo "no queue running"; exit 0; }
pkill -TERM -s "$(ps -o sid= -p "$pid" | tr -d ' ')"
echo "[$(date +%H:%M:%S)] paused (queue pid $pid)" | tee -a logs/queue_cloud.log
