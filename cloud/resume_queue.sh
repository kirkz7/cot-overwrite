#!/usr/bin/env bash
# Start or continue a cloud queue in the background (after a pause, a crash or an SSH drop).
# Finished jobs are skipped and each job skips the items it already saved.
#   usage: cloud/resume_queue.sh p0      (runs cloud/jobs_p0.txt; also p1, p2)
cd "$(dirname "$0")/.."
stage=${1:?usage: cloud/resume_queue.sh p0|p1|p2}
[ -f "cloud/jobs_$stage.txt" ] || { echo "no cloud/jobs_$stage.txt"; exit 1; }
if pgrep -f "cloud/queue_cloud.sh" > /dev/null; then echo "a queue is already running"; exit 1; fi
mkdir -p logs
setsid nohup cloud/queue_cloud.sh "cloud/jobs_$stage.txt" >> logs/queue_cloud.out 2>&1 < /dev/null &
echo "queue $stage started in the background; progress: logs/queue_cloud.log"
