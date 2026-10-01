# Application tests: (1) chat-assistant memory (LongMemEval knowledge-update), (2) newest-first logs,
# (3) agent trajectories with a retrieved stale memory, (4) the model's own long thinking.
# Resumable: a job whose "end <name> exit=0" line is already in logs\queue14.log is skipped, and every
# script skips the items already in its output file. To pause / continue: pause_queue.ps1 / resume_queue.ps1.
# Logs are written as UTF-8 by this script (not by PowerShell redirection, whose encoding differs between
# sessions in Windows PowerShell 5.1 and would mix UTF-8 and UTF-16 in one file).
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Set-Location $PSScriptRoot
$py = '.\.venv\Scripts\python.exe'
$log = 'logs\queue14.log'
function Log($msg) { Add-Content -Path $log -Value $msg -Encoding UTF8; Write-Output $msg }
Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList 'keepawake.py'
$jobs = New-Object System.Collections.ArrayList
[void]$jobs.Add(@('app_memory_read',  'run_app_memory.py read'))
[void]$jobs.Add(@('app_memory_judge', 'run_app_memory.py judge'))
[void]$jobs.Add(@('app_logs',         'run_app_logs.py --n 100'))
[void]$jobs.Add(@('app_agent',        'run_app_agent.py --n 150'))
[void]$jobs.Add(@('app_think_gen',    'run_app_thinking.py gen --budget 6144 --bs 8'))
[void]$jobs.Add(@('app_think_final',  'run_app_thinking.py final --bs 8 --answer_budget 1280'))
[void]$jobs.Add(@('app_think_probe',  'run_app_thinking.py probe --readers Qwen3-4B,Qwen3-14B'))
[void]$jobs.Add(@('app_think_stats',  'run_app_thinking.py stats'))
# mitigation baselines on LongMemEval (instruction / timeline-first / question-first / thinking mode)
[void]$jobs.Add(@('app_fix_read',     'run_app_fix.py read'))
[void]$jobs.Add(@('app_fix_think',    'run_app_fix.py think'))
[void]$jobs.Add(@('app_fix_judge',    'run_app_fix.py judge'))
# ---- LoRA fix (PREREG_TRAINING.md, data v2) ----
# 1. base-model baselines on the frozen external evals (also proves the eval scripts run end to end)
[void]$jobs.Add(@('ext_base_q4',   'run_ext_eval.py --models Qwen3-4B'))
# 2. memory probes at max_len 8192 (3 longest samples, no saving)
$P = '--data data_train/decoupled_train.jsonl --val data_train/decoupled_val.jsonl --max_len 8192 --probe'
[void]$jobs.Add(@('probe_q4',  "train_lora.py --model Qwen3-4B --out runs/probe-q4 $P"))
[void]$jobs.Add(@('probe_q17', "train_lora.py --model Qwen3-1.7B --out runs/probe-q17 $P"))
[void]$jobs.Add(@('probe_phi', "train_lora.py --model Phi-4-mini --out runs/probe-phi $P"))
# 3. training: decoupled vs chrono-only control (identical data except order), default config, final checkpoint only
$T = '--data data_train/{0}_train.jsonl --val data_train/{0}_val.jsonl --dev data_train/decoupled_dev.jsonl --rank 16 --alpha 32 --lr 1e-4 --epochs 1 --accum 8 --max_len 8192 --eval_every 200 --save_every 50 --seed {1}'
foreach ($s in 0) {
  [void]$jobs.Add(@("train_q4_dec_s$s",  ("train_lora.py --model Qwen3-4B --out runs/q4-dec-s$s "   + ($T -f 'decoupled', $s))))
  [void]$jobs.Add(@("train_q4_chr_s$s",  ("train_lora.py --model Qwen3-4B --out runs/q4-chr-s$s "   + ($T -f 'chrono', $s))))
}
[void]$jobs.Add(@('train_q17_dec_s0', ('train_lora.py --model Qwen3-1.7B --out runs/q17-dec-s0 ' + ($T -f 'decoupled', 0))))
[void]$jobs.Add(@('train_q17_chr_s0', ('train_lora.py --model Qwen3-1.7B --out runs/q17-chr-s0 ' + ($T -f 'chrono', 0))))
[void]$jobs.Add(@('train_phi_dec_s0', ('train_lora.py --model Phi-4-mini --out runs/phi-dec-s0 ' + ($T -f 'decoupled', 0))))
[void]$jobs.Add(@('train_phi_chr_s0', ('train_lora.py --model Phi-4-mini --out runs/phi-chr-s0 ' + ($T -f 'chrono', 0))))
# 4. remaining base-model baselines (all before any trained model is evaluated)
[void]$jobs.Add(@('ext_base_q17', 'run_ext_eval.py --models Qwen3-1.7B'))
[void]$jobs.Add(@('ext_base_phi', 'run_ext_eval.py --models Phi-4-mini'))
# 5. extra seeds for the main model
foreach ($s in 1, 2) {
  [void]$jobs.Add(@("train_q4_dec_s$s",  ("train_lora.py --model Qwen3-4B --out runs/q4-dec-s$s "   + ($T -f 'decoupled', $s))))
  [void]$jobs.Add(@("train_q4_chr_s$s",  ("train_lora.py --model Qwen3-4B --out runs/q4-chr-s$s "   + ($T -f 'chrono', $s))))
}
# 6. evaluations of trained models are added only after the dev-set curves confirm the configuration (no test peeking)
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
