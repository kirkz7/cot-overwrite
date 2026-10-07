# E18.1 adapter (Qwen3-4B LoRA r16, trained 2026-10-07 on the cloud L20, runs/e181-dec/final)

GitHub refuses files over 100 MB, so adapter_model.safetensors (132,187,888 bytes) is split in two parts.
Restore it into the path every script expects:

    mkdir -p runs/e181-dec/final
    git checkout origin/weights-e181 -- weights/e181-dec
    cp weights/e181-dec/adapter_config.json weights/e181-dec/README.md runs/e181-dec/final/
    cat weights/e181-dec/adapter_model.safetensors.part* > runs/e181-dec/final/adapter_model.safetensors
    sha256sum runs/e181-dec/final/adapter_model.safetensors   # must be 71423b144373dd6ea9f8613ecfd421d5b0b1aba4a08dc84da0d8b20bc1eac68a

Then use it as Qwen3-4B@runs/e181-dec/final (merged into bf16 Qwen3-4B by app_common.load_reader).
Training: train_lora.py with the CLOUD_PLAN P7 arguments (rank 16, alpha 32, lr 1e-4, 1 epoch, accum 8, max_len 8192, seed 0)
on data_train/bind4_decoupled_train.jsonl (3974 rows, 3882 used, 92 dropped > 8192 tokens; 486 steps; sha256 prefix
0B46429FD80CB99A, LF). Validation: step 300 val 95.0 / dev 85.0; final val 95.0 / dev 86.7 (step 0: 8.3 / 6.7).
The cloud's train_lora.py runs backward under the same SDPA backends as the forward (CheckpointError on L20 otherwise).
Do not retrain or modify it. runs/ is git-ignored on purpose.
