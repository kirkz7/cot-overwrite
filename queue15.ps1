# Exploration queue (EXPLORE_PLAN.md). Small, exploratory runs; same resume rules as queue14.ps1:
# a job whose "end <name> exit=0" line is already in logs\queue15.log is skipped, and every script skips
# the items already in its output file. To pause / continue: pause_queue.ps1 / resume_queue.ps1.
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
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
[void]$jobs.Add(@('e1c_judge2',  'explore_think.py judge'))
# E10: replicate Guo et al. 2026 attention routing (oracle positions) + reminder baseline; config chosen on calibration emails
[void]$jobs.Add(@('e10_calibrate', 'explore_route.py calibrate'))
[void]$jobs.Add(@('e10_run',       'explore_route.py run'))
[void]$jobs.Add(@('e10_judge',     'explore_route.py judge'))
# confirmations on other models: E7 probes on Qwen3-14B (scale), E1b prefilled timelines on Phi-4-mini (other family)
[void]$jobs.Add(@('e7_probe_q14',  'explore_probe.py run --models Qwen3-14B --n 400'))
[void]$jobs.Add(@('e1_pre_phi',    'explore_timeline.py read --models Phi-4-mini --conds pre'))
[void]$jobs.Add(@('e1_judge_phi',  'explore_timeline.py judge'))
# E5: LoRA trained only on CoT traces with step / clock-time tags (decoupled vs chronological control), tested on applications
$C = '--data data_train/cot_{0}_train.jsonl --val data_train/cot_{0}_val.jsonl --rank 16 --alpha 32 --lr 1e-4 --epochs 1 --accum 8 --max_len 2048 --eval_every 100 --save_every 50 --seed 0'
[void]$jobs.Add(@('e5_train_dec', ('train_lora.py --model Qwen3-4B --out runs/e5-cot-dec ' + ($C -f 'decoupled'))))
[void]$jobs.Add(@('e5_train_chr', ('train_lora.py --model Qwen3-4B --out runs/e5-cot-chr ' + ($C -f 'chrono'))))
$E5 = 'Qwen3-4B@runs/e5-cot-dec/final,Qwen3-4B@runs/e5-cot-chr/final'
[void]$jobs.Add(@('e5_logs',   "run_app_logs.py --n 50 --models $E5"))
[void]$jobs.Add(@('e5_agent',  "run_app_agent.py --n 75 --models $E5"))
[void]$jobs.Add(@('e5_ext',    "run_ext_eval.py --models $E5 --benches cot,mab"))
[void]$jobs.Add(@('e5_mem',    "run_app_memory.py read --models $E5"))
[void]$jobs.Add(@('e5_judge',  'run_app_memory.py judge'))
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
