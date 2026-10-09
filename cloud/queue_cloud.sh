#!/usr/bin/env bash
# Cloud (2x L20) experiment queue: the bash port of queue15.ps1 (CLOUD_PLAN.md section 4.3). Same resume rules:
# a job whose "end   <name> exit=0 (" line is already in logs/queue_cloud.log is skipped, and every script skips
# the items already in its output file.
#   usage: cloud/queue_cloud.sh cloud/jobs_p0.txt      (foreground; normally started by cloud/resume_queue.sh)
# Job files: one job per line, "<name> [VAR=value ...] <command line after python>"; blank lines and # comments are
# ignored. Qwen3-32B jobs carry COT_DEVICE_MAP=auto (two cards); 4B and the 14B judge stay on one card as on the desktop.
# Jobs with COT_ENGINE=vllm get the vLLM server started first (cloud/vllm_ctl.sh); any other job stops it first.
# The job file is re-read after every job, so jobs appended while the queue runs are picked up.
# A failed job is not retried in the same run (rerun the queue after fixing it).
set -u
cd "$(dirname "$0")/.."
JOBS=${1:?usage: cloud/queue_cloud.sh <jobs file>}
export HF_HOME=${HF_HOME:-/data/hf_cache}
export COT_DATA=${COT_DATA:-/data/datasets}
export HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 PYTHONIOENCODING=utf-8 TOKENIZERS_PARALLELISM=false
export CUDA_DEVICE_ORDER=PCI_BUS_ID   # CUDA_VISIBLE_DEVICES numbers = nvidia-smi numbers
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
    # vLLM jobs (COT_ENGINE=vllm COT_VLLM_MODEL=<hf id> [COT_VLLM_TP=n]) need the server; every other job needs free cards
    # (the server for this job's COT_VLLM_URL port and CUDA_VISIBLE_DEVICES; other queues' servers are left alone)
    vport=$(sed -n 's/.*COT_VLLM_URL=[^ ]*:\([0-9]*\).*/\1/p' <<< "$nextcmd_env"); vport=${vport:-8000}
    if [[ " $nextcmd_env" == *" COT_ENGINE=vllm "* ]]; then
        vm=$(sed -n 's/.*COT_VLLM_MODEL=\([^ ]*\).*/\1/p' <<< "$nextcmd_env"); vtp=$(sed -n 's/.*COT_VLLM_TP=\([^ ]*\).*/\1/p' <<< "$nextcmd_env")
        env $nextcmd_env cloud/vllm_ctl.sh up "$vm" "${vtp:-2}" >> "logs/$next.log" 2>&1
    elif [ -f "logs/vllm_server_$vport.pid" ]; then
        env $nextcmd_env cloud/vllm_ctl.sh down >> "logs/$next.log" 2>&1
    fi
    rc=$?
    [ $rc -eq 0 ] && { env $nextcmd_env $PY $nextcmd >> "logs/$next.log" 2>&1; rc=$?; }
    log "[$(now)] end   $next exit=$rc ($(( ($(date +%s) - t) / 60 )) min)"
done
log "[$(now)] queue done $JOBS"
