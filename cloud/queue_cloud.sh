#!/usr/bin/env bash
# Cloud (2x L20) experiment queue: the bash port of queue15.ps1 (CLOUD_PLAN.md section 4.3). Same resume rules:
# a job whose "end   <name> exit=0 (" line is already in logs/queue_cloud.log is skipped, and every script skips
# the items already in its output file.
#   usage: cloud/queue_cloud.sh cloud/jobs_p0.txt      (foreground; normally started by cloud/resume_queue.sh)
# Job files: one job per line, "<name> [VAR=value ...] <command line after python>"; blank lines and # comments are
# ignored. Qwen3-32B jobs carry COT_DEVICE_MAP=auto (two cards); 4B and the 14B judge stay on one card as on the desktop.
# The job file is re-read after every job, so jobs appended while the queue runs are picked up.
# A failed job is not retried in the same run (rerun the queue after fixing it).
set -u
cd "$(dirname "$0")/.."
JOBS=${1:?usage: cloud/queue_cloud.sh <jobs file>}
export HF_HOME=${HF_HOME:-/data/hf_cache}
export COT_DATA=${COT_DATA:-/data/datasets}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
PY=${COT_PY:-.venv/bin/python}
LOG=logs/queue_cloud.log
mkdir -p logs
log() { echo "$1" | tee -a "$LOG"; }
now() { date +%H:%M:%S; }
done_ok() { [ -f "$LOG" ] && grep -qF "end   $1 exit=0 (" "$LOG"; }

declare -A tried=()
log "[$(now)] queue start $JOBS"
while :; do
    next=""
    while read -r name cmd; do
        [ -z "$name" ] || [[ $name == \#* ]] && continue
        [ -n "${tried[$name]:-}" ] && continue
        if done_ok "$name"; then
            log "[$(now)] skip  $name (already finished)"; tried[$name]=1; continue
        fi
        next=$name; nextcmd_env=""; nextcmd=$cmd
        while [[ ${nextcmd%% *} == *=* ]]; do nextcmd_env+="${nextcmd%% *} "; nextcmd=${nextcmd#* }; done
        break
    done < "$JOBS"
    [ -z "$next" ] && break
    tried[$next]=1
    t=$(date +%s)
    log "[$(now)] start $next"
    env $nextcmd_env $PY $nextcmd >> "logs/$next.log" 2>&1
    rc=$?
    log "[$(now)] end   $next exit=$rc ($(( ($(date +%s) - t) / 60 )) min)"
done
log "[$(now)] queue done $JOBS"
