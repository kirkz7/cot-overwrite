# E18.1 training-data inputs made on the cloud (git-ignored on main)

selfdistill_train.jsonl: `python gen_selfdistill.py --n 600 --engine vllm` on the cloud (2x L20), 10-07 07:57-08:04,
after the source filter fix of 10-07 (`ifdata`, `math-grade` excluded; see CLOUD_NOTEBOOK.md on branch cloud-l20).
600 prompts from allenai/tulu-3-sft-mixture @ b14afda6..., 15 sources x 40; replies by the base Qwen3-4B (vLLM 0.31.0, seed 1818),
half thinking on (temperature 0.6, top-p 0.95, top-k 20, <= 3072 tokens), half off (0.7 / 0.8 / 20, <= 1024 tokens).
Kept 474 rows (thinking on 221 / off 253); dropped 126 unfinished, 0 too long.

Restore into the path gen_bind_data_v4.py reads:

    mkdir -p data_train
    git show origin/weights-e181:data/selfdistill_train.jsonl > data_train/selfdistill_train.jsonl
    sha256sum data_train/selfdistill_train.jsonl   # must be 78a512959c2b2d7aeedada9ea7a3080408dcba8e85961c61cd5a59566b12f5bb

The file is written with LF line ends (Linux). Checked out on Windows with autocrlf it becomes CRLF; then the sha256 is
that of the CRLF version (prefix CFCA555F7CBB8942). gen_bind_data_v4.py on the cloud then gives bind4_decoupled_train.jsonl
with sha256 prefix 0B46429FD80CB99A (LF).
