# Exploration queue (EXPLORE_PLAN.md). Small, exploratory runs; same resume rules as queue14.ps1:
# a job whose "end <name> exit=0" line is already in logs\queue15.log is skipped, and every script skips
# the items already in its output file. To pause / continue: pause_queue.ps1 / resume_queue.ps1.
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:HF_DATASETS_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Set-Location $PSScriptRoot
$py = '.\.venv\Scripts\python.exe'
$log = 'logs\queue15.log'
function Log($msg) { Add-Content -Path $log -Value $msg -Encoding UTF8; Write-Output $msg }
$started = Get-Date
$ka = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'keepawake\.py' -and $_.CommandLine -notmatch 'Get-CimInstance' })
if ($ka.Count -eq 0) { Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList 'keepawake.py' }
$jobs = New-Object System.Collections.ArrayList
# E1b: prefilled timelines on the base models (judged right away), then E2: the decoupled / control LoRA
[void]$jobs.Add(@('e1_pre_q4',   'explore_timeline.py read --models Qwen3-4B --conds pre'))
[void]$jobs.Add(@('e1_judge_q4', 'explore_timeline.py judge'))
[void]$jobs.Add(@('e1_pre_q14',  'explore_timeline.py read --models Qwen3-14B --conds pre'))
[void]$jobs.Add(@('e1_judge_q14', 'explore_timeline.py judge'))
[void]$jobs.Add(@('e2_dec_chr',  'explore_timeline.py read --models Qwen3-4B@runs/q4-dec-s0/final,Qwen3-4B@runs/q4-chr-s0/final --conds pre,gen'))
[void]$jobs.Add(@('e2_judge',    'explore_timeline.py judge'))
# E1d: same timeline prompt, but the list must be written oldest first (exploits reading by position)
[void]$jobs.Add(@('e1d_sorted_q4', 'explore_timeline.py read --models Qwen3-4B --conds sorted'))
[void]$jobs.Add(@('e1d_judge',     'explore_timeline.py judge'))
# E1c: thinking mode with the thinking text saved (LongMemEval + our email threads)
[void]$jobs.Add(@('e1c_think_q4',  'explore_think.py gen --models Qwen3-4B'))
[void]$jobs.Add(@('e1c_judge',     'explore_think.py judge'))
# E3: the recency heads found on CoT traces (Exp 7 / 14), zeroed on application tasks
[void]$jobs.Add(@('e3_heads_run',   'explore_heads.py run'))
[void]$jobs.Add(@('e3_heads_judge', 'explore_heads.py judge'))
# E7: linear probes - does the model internally know which value is newer? (base, decoupled, control)
[void]$jobs.Add(@('e7_probe_run',   'explore_probe.py run --n 400'))
# E4: recency or primacy as the CoT trace grows to ~19k tokens (link to Guo et al. 2026)
[void]$jobs.Add(@('e4_primacy',     'explore_primacy.py run --n 50'))
# ---- 10-03 afternoon (queued while the user is away; thresholds in EXPLORE_PLAN.md) ----
# E1d/E2 measurement fix: a second step "Answer:" for outputs without an answer line, then re-judge
[void]$jobs.Add(@('e1d_repair',  'explore_timeline.py repair --models Qwen3-4B,Qwen3-4B@runs/q4-dec-s0/final,Qwen3-4B@runs/q4-chr-s0/final'))
[void]$jobs.Add(@('e1d_judge2',  'explore_timeline.py judge'))
# E1c swap (thinking of one order after the input of the other) and length control (padded email threads)
[void]$jobs.Add(@('e1c_swap',    'explore_think.py swap --models Qwen3-4B'))
[void]$jobs.Add(@('e1c_pad',     'explore_think.py gen --pad --models Qwen3-4B'))
[void]$jobs.Add(@('e11_notes',   'explore_think.py notes --models Qwen3-4B'))
[void]$jobs.Add(@('e1c_judge2',  'explore_think.py judge'))
# E10 (diagnostic only, not a fix): Guo et al. 2026 attention routing with oracle positions + reminder baseline
[void]$jobs.Add(@('e10_calibrate', 'explore_route.py calibrate'))
[void]$jobs.Add(@('e10_run',       'explore_route.py run'))
[void]$jobs.Add(@('e10_judge',     'explore_route.py judge'))
# confirmations on other models: E7 probes on Qwen3-14B (scale), E1b prefilled timelines on Phi-4-mini (other family)
[void]$jobs.Add(@('e7_probe_q14',  'explore_probe.py run --models Qwen3-14B --n 400'))
[void]$jobs.Add(@('e1_pre_phi',    'explore_timeline.py read --models Phi-4-mini --conds pre'))
[void]$jobs.Add(@('e1_judge_phi',  'explore_timeline.py judge'))
# E12: LoRA trained to write time-organized reasoning, then answer (same inputs as v2; decoupled vs chronological control)
$R = '--data data_train/reason_{0}_train.jsonl --val data_train/reason_{0}_val.jsonl --dev data_train/decoupled_dev.jsonl --rank 16 --alpha 32 --lr 1e-4 --epochs 1 --accum 8 --max_len 8192 --eval_every 300 --save_every 50 --val_n 60 --dev_n 60 --seed 0'
[void]$jobs.Add(@('e12_train_dec', ('train_lora.py --model Qwen3-4B --out runs/e12-dec ' + ($R -f 'decoupled'))))
[void]$jobs.Add(@('e12_train_chr', ('train_lora.py --model Qwen3-4B --out runs/e12-chr ' + ($R -f 'chrono'))))
[void]$jobs.Add(@('e12_eval',      'explore_reason_eval.py run --models Qwen3-4B@runs/e12-dec/final,Qwen3-4B@runs/e12-chr/final,Qwen3-4B,Qwen3-4B@runs/q4-dec-s0/final'))
[void]$jobs.Add(@('e12_judge',     'explore_reason_eval.py judge'))
# long-dialogue test sets beyond LongMemEval (base models only; frozen as held-out tests)
[void]$jobs.Add(@('lc_q4',    'explore_longconv.py run --models Qwen3-4B'))
[void]$jobs.Add(@('lc_phi',   'explore_longconv.py run --models Phi-4-mini'))
[void]$jobs.Add(@('lc_judge', 'explore_longconv.py judge'))
# ---- overnight 10-04 (user asked to use the GPU fully; thresholds in EXPLORE_PLAN.md) ----
# E2 on Phi: the pre-registered Phi LoRAs (decoupled / control), prefilled timelines
[void]$jobs.Add(@('e2_phi',       'explore_timeline.py read --models Phi-4-mini@runs/phi-dec-s0/final,Phi-4-mini@runs/phi-chr-s0/final --conds pre'))
[void]$jobs.Add(@('e2_phi_judge', 'explore_timeline.py judge'))
# E13: train fact-time binding on long block-dated documents (no chat formats); test on the frozen chat sets
$B = '--data data_train/bind_{0}_train.jsonl --val data_train/bind_{0}_val.jsonl --dev data_train/bind_decoupled_dev.jsonl --rank 16 --alpha 32 --lr 1e-4 --epochs 1 --accum 8 --max_len 8192 --eval_every 300 --save_every 50 --val_n 60 --dev_n 60 --seed 0'
[void]$jobs.Add(@('e13_train_dec', ('train_lora.py --model Qwen3-4B --out runs/e13-dec ' + ($B -f 'decoupled'))))
[void]$jobs.Add(@('e13_dec_longconv', 'explore_longconv.py run --budget 320 --models Qwen3-4B@runs/e13-dec/final'))
[void]$jobs.Add(@('e13_dec_reason',   'explore_reason_eval.py run --models Qwen3-4B@runs/e13-dec/final'))
[void]$jobs.Add(@('e13_dec_judge1',   'explore_longconv.py judge'))
[void]$jobs.Add(@('e13_dec_judge2',   'explore_reason_eval.py judge'))
# E14: thinking mode with the decoupled / control answer-only LoRA (does the fix reach the model's own thinking?)
[void]$jobs.Add(@('e14_think',  'explore_think.py gen --models Qwen3-4B@runs/q4-dec-s0/final,Qwen3-4B@runs/q4-chr-s0/final'))
[void]$jobs.Add(@('e14_judge',  'explore_think.py judge'))
# judge the E13-control ConvoMem-long rows already generated (primary E13 threshold) before the long PersonaMem part
[void]$jobs.Add(@('e13_chr_judge0', 'explore_longconv.py judge'))
# PM-diag: why no order effect on PersonaMem (no-history controls + the other 5 question types), base models
[void]$jobs.Add(@('pmdiag', 'explore_pmdiag.py run --models Qwen3-4B,Phi-4-mini'))
# External memory sets (MemConflict, MemoryAgentBench-CR, LoCoMo) x (chrono, rev, BM25 retrieval order), base 4B first
[void]$jobs.Add(@('ext_q4',     'explore_extmem.py run --models Qwen3-4B'))
[void]$jobs.Add(@('ext_judge1', 'explore_extmem.py judge'))
# MemConflict fix test moved up (user, 10-04 21:20): E13 dec / control, reasoning budget 320
[void]$jobs.Add(@('ext_e13_mc', 'explore_extmem.py run --tasks memconf --budget 320 --models Qwen3-4B@runs/e13-dec/final,Qwen3-4B@runs/e13-chr/final'))
[void]$jobs.Add(@('ext_judge_mc', 'explore_extmem.py judge'))
# E16 (review 10-04): MemConflict items whose update session restates the old value, base 4B (boundary control)
[void]$jobs.Add(@('ext_rs_q4', 'explore_extmem.py run --tasks memconf_rs --models Qwen3-4B'))
[void]$jobs.Add(@('ext_judge_rs', 'explore_extmem.py judge'))
# E17 (10-05, user: retrain): binding data with change questions + format instructions; judge parse v2 (whole answer)
[void]$jobs.Add(@('ext_judge_v2', 'explore_extmem.py judge --parse v2'))
[void]$jobs.Add(@('e17_train', 'train_lora.py --model Qwen3-4B --out runs/e17-dec --data data_train/bind2_decoupled_train.jsonl --val data_train/bind2_decoupled_val.jsonl --dev data_train/bind2_decoupled_dev.jsonl --rank 16 --alpha 32 --lr 1e-4 --epochs 1 --accum 8 --max_len 8192 --eval_every 300 --save_every 50 --val_n 60 --dev_n 60 --seed 0'))
[void]$jobs.Add(@('e17_format', 'explore_format_eval.py run --models Qwen3-4B@runs/e17-dec/final,Qwen3-4B@runs/e13-dec/final'))
[void]$jobs.Add(@('e17_mc', 'explore_extmem.py run --tasks memconf --budget 320 --models Qwen3-4B@runs/e17-dec/final'))
[void]$jobs.Add(@('e17_mc_judge', 'explore_extmem.py judge --parse v2'))
[void]$jobs.Add(@('e17_longconv', 'explore_longconv.py run --budget 320 --models Qwen3-4B@runs/e17-dec/final'))
[void]$jobs.Add(@('e17_lc_judge', 'explore_longconv.py judge'))
# E18 (10-06 01:30, user: training first): dated list inside Qwen3's thinking block; evaluated with thinking on AND off
[void]$jobs.Add(@('e18_train', 'train_lora.py --model Qwen3-4B --out runs/e18-dec --data data_train/bind3_decoupled_train.jsonl --val data_train/bind3_decoupled_val.jsonl --dev data_train/bind3_decoupled_dev.jsonl --rank 16 --alpha 32 --lr 1e-4 --epochs 1 --accum 8 --max_len 8192 --eval_every 300 --save_every 50 --val_n 60 --dev_n 60 --seed 0'))
[void]$jobs.Add(@('e18_format',    'explore_format_eval.py run --data bind3 --modes think,direct --models Qwen3-4B@runs/e18-dec/final'))
[void]$jobs.Add(@('e18_format_s',  'explore_format_eval.py run --data bind3 --modes think --models Qwen3-4B@runs/e18-dec/final'))   # thinking: Qwen3 sampling (10-06)
[void]$jobs.Add(@('e18_lc_think',  'explore_longconv.py run --think --budget 1024 --models Qwen3-4B@runs/e18-dec/final'))
[void]$jobs.Add(@('e18_lc_direct', 'explore_longconv.py run --models Qwen3-4B@runs/e18-dec/final'))
[void]$jobs.Add(@('e18_lc_judge',  'explore_longconv.py judge'))
[void]$jobs.Add(@('e18_mc_think',  'explore_extmem.py run --tasks memconf --think --budget 1024 --models Qwen3-4B@runs/e18-dec/final'))
[void]$jobs.Add(@('e18_mc_judge',  'explore_extmem.py judge --parse v2'))
[void]$jobs.Add(@('e18_reason',    'explore_reason_eval.py run --think --models Qwen3-4B@runs/e18-dec/final'))
[void]$jobs.Add(@('e18_reason_d',  'explore_reason_eval.py run --models Qwen3-4B@runs/e18-dec/final'))
[void]$jobs.Add(@('e18_reason_j',  'explore_reason_eval.py judge'))
[void]$jobs.Add(@('e18_mc_direct', 'explore_extmem.py run --tasks memconf --models Qwen3-4B@runs/e18-dec/final'))
[void]$jobs.Add(@('e18_mc_judge2', 'explore_extmem.py judge --parse v2'))
# E18 design diagnostic (10-05 22:30): is writing the dated list needed? E13b (answer-only) vs E13 (list); E17 on the same sets
[void]$jobs.Add(@('diag_e13b_lc',  'explore_longconv.py run --budget 320 --tasks convo_long --models Qwen3-4B@runs/e13b-dec/final'))
[void]$jobs.Add(@('diag_reason',   'explore_reason_eval.py run --models Qwen3-4B@runs/e13b-dec/final,Qwen3-4B@runs/e17-dec/final'))
[void]$jobs.Add(@('diag_judge1',   'explore_longconv.py judge'))
[void]$jobs.Add(@('diag_judge2',   'explore_reason_eval.py judge'))
# no-harm suite: moved to the cloud 10-06 (CLOUD_PLAN P6: base + E18); E17 and the short-dialogue LoRAs dropped (superseded by E18)
# E15 (review 10-04): earliest-value questions separate 'later text read as later time' from mechanical recency
[void]$jobs.Add(@('e15_rule', 'explore_order_rule.py run --models Qwen3-4B,Phi-4-mini'))
# E13 control
[void]$jobs.Add(@('e13_train_chr', ('train_lora.py --model Qwen3-4B --out runs/e13-chr ' + ($B -f 'chrono'))))
[void]$jobs.Add(@('e13_chr_longconv', 'explore_longconv.py run --budget 320 --models Qwen3-4B@runs/e13-chr/final'))
[void]$jobs.Add(@('e13_chr_reason',   'explore_reason_eval.py run --models Qwen3-4B@runs/e13-chr/final'))
[void]$jobs.Add(@('e13_chr_judge1',   'explore_longconv.py judge'))
[void]$jobs.Add(@('e13_chr_judge2',   'explore_reason_eval.py judge'))
# E13b: trained (runs/e13b-dec kept) but its tests were dropped 10-05: E13 superseded by E17 (value-only answers)
[void]$jobs.Add(@('ext_phi',    'explore_extmem.py run --models Phi-4-mini'))
[void]$jobs.Add(@('ext_judge2', 'explore_extmem.py judge'))
[void]$jobs.Add(@('e13b_train', 'train_lora.py --model Qwen3-4B --out runs/e13b-dec --data data_train/bindans_decoupled_train.jsonl --val data_train/bindans_decoupled_val.jsonl --dev data_train/bind_decoupled_dev.jsonl --rank 16 --alpha 32 --lr 1e-4 --epochs 1 --accum 8 --max_len 8192 --eval_every 300 --save_every 50 --val_n 60 --dev_n 60 --seed 0'))
# (10-06) Qwen3-14B long-dialogue base test moved to the cloud (CLOUD_PLAN P1); the 5080 trains and explores 4B only
foreach ($j in $jobs) {
    $name = $j[0]; $cmd = $j[1]
    if ((Test-Path $log) -and (Select-String -Path $log -SimpleMatch "end   $name exit=0 (" -Quiet)) {
        Log "[$((Get-Date).ToString('HH:mm:ss'))] skip  $name (already finished)"
        continue
    }
    $t = Get-Date
    Log "[$($t.ToString('HH:mm:ss'))] start $name"
    cmd /c "$py $cmd >> logs\$name.log 2>&1"
    Log "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
# jobs added to this file while it was running are picked up here (finished jobs are skipped)
if ((Get-Item $PSCommandPath).LastWriteTime -gt $started) { & $PSCommandPath }
