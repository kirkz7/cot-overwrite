# Desktop reference outputs for the P0 port check (CLOUD_PLAN.md section 4.5)

Do NOT copy these into results/: every script skips work already in its results/ file, so a copy there would make the
fresh run skip everything. Run fresh into results/, then `python compare_reference.py logs` and `... e15`.

| file | made by | rows | sha256 |
|---|---|---|---|
| app_logs_Qwen3-4B.jsonl | `run_app_logs.py --n 100 --models Qwen3-4B` (10-01) | 1800 | 8f3b353fac29401ea6c2351b8e48a9f3bba82b664e985d934011e4d924d735c7 |
| e15_rule_Qwen3-4B.jsonl | `explore_order_rule.py run --models Qwen3-4B` (10-05) | 900 | 96803347cc2132d5fc77b22ea3a0a1578e6f46bd38a3c97d7790f617cb78d115 |

Both are synthetic data only (no held-out or LongMemEval text). Desktop environment: RTX 5080 16 GB, driver 616.92,
Windows 11, Python 3.12.14, torch 2.11.0+cu128, transformers 5.17.0, peft 0.21.2; Qwen3-4B bf16 on one GPU, greedy.
Code affecting these runs is unchanged since (app_common.py only gained answer_text on 10-05).
