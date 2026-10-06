# E18 adapter (Qwen3-4B LoRA r16, trained 2026-10-06 on the desktop, runs/e18-dec/final)

GitHub refuses files over 100 MB, so adapter_model.safetensors (132,187,888 bytes) is split in two parts.
Restore it into the path every script expects:

    mkdir -p runs/e18-dec/final
    cp weights/e18-dec/adapter_config.json weights/e18-dec/README.md runs/e18-dec/final/
    cat weights/e18-dec/adapter_model.safetensors.part* > runs/e18-dec/final/adapter_model.safetensors
    sha256sum runs/e18-dec/final/adapter_model.safetensors   # must be 45f05d3d41fbd9fc3a4428e5e6d0eafa91a5fff3345e27af2712a348901e91e2

Then use it as Qwen3-4B@runs/e18-dec/final (merged into bf16 Qwen3-4B by app_common.load_reader).
Do not retrain or modify it. runs/ is git-ignored on purpose.
