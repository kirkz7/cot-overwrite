# E18.1 recipe on Qwen3-8B (LoRA r16, trained 2026-10-08 on the cloud L20, runs/e181-q8/final)

adapter_model.safetensors (174,655,536 bytes) is split in parts for GitHub's 100 MB limit:

    mkdir -p runs/e181-q8/final
    git checkout origin/weights-e181 -- weights/e181-q8
    cp weights/e181-q8/adapter_config.json weights/e181-q8/README.md runs/e181-q8/final/
    cat weights/e181-q8/adapter_model.safetensors.part* > runs/e181-q8/final/adapter_model.safetensors
    sha256sum runs/e181-q8/final/adapter_model.safetensors   # must be 1630e06a0eafb207d95037e3dd67db7f50797ea2d3837fc013ff6376e617426e

Use as Qwen3-8B-bf16@runs/e181-q8/final (bf16 Qwen3-8B, MODELS key on branch cloud-l20).
Data: bind4q8_decoupled_train.jsonl (3963 rows: the 4B E18.1 construction + 463 replies self-distilled from Qwen3-8B;
sha256 prefix 117035D09E739ACB, LF), same hyperparameters as E18.1. 484 steps (3867 rows used). Validation: step 0
16.7 / 11.7, step 300 95.0 / 91.7, final 96.7 / 90.0 (val / dev).
