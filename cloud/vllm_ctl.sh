#!/usr/bin/env bash
# Start / stop the vLLM server that COT_ENGINE=vllm jobs talk to (vllm_client.py). vLLM lives in its own venv
# ($VLLM_VENV, default /data/vllm-venv) because it pins its own torch; the experiment venv stays at the desktop versions.
#   cloud/vllm_ctl.sh up Qwen/Qwen3-32B [tp]   (no-op if that model is already served; tp default: 2)
#   cloud/vllm_ctl.sh down
# One server per port: COT_VLLM_URL (default http://127.0.0.1:8000) picks the port, CUDA_VISIBLE_DEVICES the cards,
# so two single-card servers (e.g. 4B on GPU0 :8000, 14B on GPU1 :8001) can run side by side.
# Fixed server settings (CLOUD_NOTEBOOK.md "vLLM"): bf16, max_model_len 43008 (4 ToT prompts reach 42k tokens; HF on the
# desktop ran them past the 40960 trained length, so vLLM is allowed to as well), the model's generation_config ignored,
# seed 0. The client sends one request at a time.
set -u
export CUDA_DEVICE_ORDER=PCI_BUS_ID
cd "$(dirname "$0")/.."
VENV=${VLLM_VENV:-/data/vllm-venv}
URL=${COT_VLLM_URL:-http://127.0.0.1:8000}
PORT=${URL##*:}
PIDF=logs/vllm_server_$PORT.pid
SLOG=logs/vllm_server_$PORT.log
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
    # a server on this port that this script did not start (no pid file): stop it too
    for p in $(ss -ltnpH "sport = :$PORT" 2>/dev/null | grep -o 'pid=[0-9]*' | cut -d= -f2 | sort -u); do
        kill -TERM -- "-$(ps -o pgid= -p "$p" | tr -d ' ')" 2>/dev/null || kill -TERM "$p" 2>/dev/null
    done
    for _ in $(seq 30); do [ -z "$(served)" ] && break; sleep 2; done
    # wait until this server's cards are free again (the next job may load an HF model)
    gpus=${CUDA_VISIBLE_DEVICES:-}
    for _ in $(seq 60); do
        used=$(nvidia-smi ${gpus:+-i $gpus} --query-gpu=memory.used --format=csv,noheader,nounits | sort -n | tail -1)
        [ "$used" -lt 2000 ] && break; sleep 2
    done
    echo "[$(date +%H:%M:%S)] vllm :$PORT down"
}

up() {
    model=$1; tp=${2:-2}
    [ "$(served)" = "$model" ] && { echo "[$(date +%H:%M:%S)] vllm :$PORT already serving $model"; return 0; }
    down
    echo "[$(date +%H:%M:%S)] vllm :$PORT up $model tp=$tp gpus=${CUDA_VISIBLE_DEVICES:-all}" | tee -a "$SLOG"
    # flashinfer compiles kernels at start-up: it needs the venv's ninja and the CUDA 13.0 toolkit (torch is cu130)
    PATH="$VENV/bin:/usr/local/cuda-13.0/bin:$PATH" CUDA_HOME=/usr/local/cuda-13.0 HF_HUB_OFFLINE=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 setsid nohup \
        "$VENV/bin/vllm" serve "$model" --tensor-parallel-size "$tp" --dtype bfloat16 \
        --max-model-len 43008 --generation-config vllm --seed 0 --gpu-memory-utilization 0.88 \
        --host 127.0.0.1 --port "$PORT" >> "$SLOG" 2>&1 < /dev/null &
    echo $! > "$PIDF"
    for _ in $(seq 360); do
        [ "$(served)" = "$model" ] && { echo "[$(date +%H:%M:%S)] vllm :$PORT ready"; return 0; }
        kill -0 "$(cat "$PIDF")" 2>/dev/null || { echo "vllm server died, see $SLOG"; return 1; }
        sleep 5
    done
    echo "vllm server not ready after 30 min"; return 1
}

case "${1:-}" in
    up) up "${2:?model}" "${3:-2}" ;;
    down) down ;;
    *) echo "usage: cloud/vllm_ctl.sh up <hf id> [tp] | down"; exit 1 ;;
esac
