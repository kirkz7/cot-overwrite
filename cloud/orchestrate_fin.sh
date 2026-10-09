#!/usr/bin/env bash
# CLOUD_PLAN P8 (10-08 review): the last night of the lease (Beijing time, 10-09; lease ends ~07:30).
#   GPU0: after jobs_c -> jobs_fin0 (14B thinking on, MemConflict chrono + rev, + judge); while GPU1 is still busy
#         before 05:00 -> jobs_fin0x (14B retr, + judge)
#   both: GPU1 free (jobs_lme8 over) before 05:00 -> jobs_fin2 (32B thinking on, chrono + rev); jobs_fin2x (32B retr)
#         only if it can start before 05:20. GPU1 still busy at 05:00 -> no 32B (pausing LongMemEval-8B is the user's call)
#   generation time left -> jobs_fin0, jobs_fin0x (resumed); 06:05 every generation stops -> jobs_finj (judge, parse v2)
#   06:30 all GPU work over (jobs_lme8 too, if it still runs: its saved rows are kept) -> 06:30-07:30 upload window.
# Rule: when GPU1 frees, a GPU0 queue still running gets at most 15 more minutes (never past 05:00), then both cards
# go to 32B. Every run takes its items in a fixed random order, so a stop leaves a random subsample of whole items.
# start once:  setsid nohup cloud/orchestrate_fin.sh >> logs/queue_cloud.out 2>&1 < /dev/null &
cd "$(dirname "$0")/.."
DAY=${FIN_DAY:-2026-10-09}
at() { TZ=Asia/Shanghai date -d "$DAY $1" +%s; }
TP2_LATEST=${FIN_TP2_LATEST:-$(at 05:00)}; X_LATEST=${FIN_X_LATEST:-$(at 05:20)}
GEN_STOP=${FIN_GEN_STOP:-$(at 06:05)}; HARD_STOP=${FIN_HARD_STOP:-$(at 06:30)}
POLL=${FIN_POLL:-30}; YIELD=${FIN_YIELD:-900}   # (FIN_* overrides: dry runs only)
now() { date +%s; }
log() { echo "[$(date +%H:%M:%S)] orchestrate_fin: $*" | tee -a logs/queue_cloud.log; }
running() { pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" > /dev/null; }
stop_q() { local pid; pid=$(pgrep -f "[q]ueue_cloud.sh cloud/jobs_$1.txt" | head -1)
    [ -n "$pid" ] && pkill -TERM -s "$(ps -o sid= -p "$pid" | tr -d ' ')" && log "stopped queue $1"; }
start_q() { log "start queue $1"; cloud/resume_queue.sh "$1" > /dev/null; sleep 5; }
run_until() { start_q "$1"; while running "$1"; do [ "$(now)" -ge "$2" ] && { stop_q "$1"; break; }; sleep "$POLL"; done; }
g1_free=""
mark_g1() { if [ -z "$g1_free" ] && ! running lme8; then g1_free=$(now); log "GPU1 free"; fi; }
tp2_ok() { [ -n "$g1_free" ] && [ "$g1_free" -lt "$TP2_LATEST" ]; }
run_g0() {   # a single-card queue on GPU0 that makes way for 32B (see the rule above); stops at 06:05 at the latest
    start_q "$1"
    while running "$1"; do
        mark_g1
        if tp2_ok; then
            local end=$((g1_free + YIELD)); [ "$end" -gt "$TP2_LATEST" ] && end=$TP2_LATEST
            [ "$(now)" -ge "$end" ] && { stop_q "$1"; log "$1 makes way: both cards go to 32B"; break; }
        fi
        [ "$(now)" -ge "$GEN_STOP" ] && { stop_q "$1"; break; }
        sleep "$POLL"
    done
}

log "waiting for GPU0 (jobs_c)"
while running c; do sleep "$POLL"; done
mark_g1
[ "$(now)" -lt "$GEN_STOP" ] && run_g0 fin0
mark_g1
[ -z "$g1_free" ] && [ "$(now)" -lt "$TP2_LATEST" ] && run_g0 fin0x   # GPU1 still busy: 14B retr meanwhile
while [ -z "$g1_free" ] && [ "$(now)" -lt "$TP2_LATEST" ]; do sleep "$POLL"; mark_g1; done
if tp2_ok && [ "$(now)" -lt "$GEN_STOP" ]; then
    run_until fin2 "$GEN_STOP"
    if [ "$(now)" -lt "$X_LATEST" ]; then run_until fin2x "$GEN_STOP"; else log "no time for jobs_fin2x"; fi
else
    log "GPU1 not free by 05:00: 32B skipped"
fi
for q in fin0 fin0x; do   # generation time left: finish 14B chrono + rev, then retr (each a no-op once done)
    [ "$(now)" -lt "$GEN_STOP" ] && run_until "$q" "$GEN_STOP"
done
[ "$(now)" -lt "$HARD_STOP" ] && run_until finj "$HARD_STOP"
while running lme8 && [ "$(now)" -lt "$HARD_STOP" ]; do sleep "$POLL"; done   # the user's LongMemEval-8B runs until 06:30
if running lme8; then stop_q lme8; log "jobs_lme8 stopped for the upload (saved rows are kept; judge the rest on the desktop)"; fi
for p in 8000 8001; do COT_VLLM_URL=http://127.0.0.1:$p cloud/vllm_ctl.sh down > /dev/null 2>&1; done
log "GPU work over: compute stats (diag_think_scale.py) and upload (CLOUD_PLAN P8, 06:30-07:30)"
