#!/usr/bin/env bash
# Start / stop the vLLM server that COT_ENGINE=vllm jobs talk to (vllm_client.py). vLLM lives in its own venv
# ($VLLM_VENV, default /data/vllm-venv) because it pins its own torch; the experiment venv stays at the desktop versions.
#   cloud/vllm_ctl.sh up Qwen/Qwen3-32B [tp]   (no-op if that model is already served; tp default: 2)
#   cloud/vllm_ctl.sh down
# Fixed server settings (part of the pre-registered engine, CLOUD_NOTEBOOK.md "vLLM"): bf16, max_model_len 40960,
# the model's generation_config ignored, seed 0. The client sends one request at a time.
set -u
cd "$(dirname "$0")/.."
VENV=${VLLM_VENV:-/data/vllm-venv}
URL=${COT_VLLM_URL:-http://127.0.0.1:8000}
PIDF=logs/vllm_server.pid
mkdir -p logs

served() { curl -sf "$URL/v1/models" 2>/dev/null | grep -o '"id":"[^"]*"' | head -1 | cut -d'"' -f4; }

down() {
    if [ -f "$PIDF" ]; then
        pid=$(cat "$PIDF")
        kill -TERM -- "-$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
        for _ in $(seq 60); do kill -0 "$pid" 2>/dev/null || break; sleep 2; done
        kill -KILL -- "-$pid" 2>/dev/null
        rm -f "$PIDF"
    fi
    # wait until the cards are free again (the next job may load an HF model)
    for _ in $(seq 60); do
        used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        [ "$used" -lt 2000 ] && break; sleep 2
    done
    echo "[$(date +%H:%M:%S)] vllm down"
}

up() {
    model=$1; tp=${2:-2}
    [ "$(served)" = "$model" ] && { echo "[$(date +%H:%M:%S)] vllm already serving $model"; return 0; }
    down
    echo "[$(date +%H:%M:%S)] vllm up $model tp=$tp" | tee -a logs/vllm_server.log
    HF_HUB_OFFLINE=1 setsid nohup "$VENV/bin/vllm" serve "$model" --tensor-parallel-size "$tp" --dtype bfloat16 \
        --max-model-len 40960 --generation-config vllm --seed 0 --gpu-memory-utilization 0.88 \
        --host 127.0.0.1 --port "${URL##*:}" >> logs/vllm_server.log 2>&1 < /dev/null &
    echo $! > "$PIDF"
    for _ in $(seq 360); do
        [ "$(served)" = "$model" ] && { echo "[$(date +%H:%M:%S)] vllm ready"; return 0; }
        kill -0 "$(cat "$PIDF")" 2>/dev/null || { echo "vllm server died, see logs/vllm_server.log"; return 1; }
        sleep 5
    done
    echo "vllm server not ready after 30 min"; return 1
}

case "${1:-}" in
    up) up "${2:?model}" "${3:-2}" ;;
    down) down ;;
    *) echo "usage: cloud/vllm_ctl.sh up <hf id> [tp] | down"; exit 1 ;;
esac
