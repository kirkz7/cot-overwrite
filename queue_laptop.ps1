# Laptop queue (RTX 4060 Laptop, 8 GB). Only jobs that fit in 8 GB: the pending Qwen3-1.7B part of the
# pre-registered confirmation stage (queue14 steps 4 / 7: base model + seed-0 decoupled / control LoRA).
# Generation only. The LongMemEval judge needs Qwen3-14B 4-bit (about 9 GB), so it runs on the desktop after
# the results are copied back (see LAPTOP.md). Commands are identical to queue14, so outputs are interchangeable.
# Same resume rules as the other queues: finished jobs (logs\queue_laptop.log) are skipped, and every script
# skips the items already in its output file. Pause / continue: pause_queue.ps1 / resume_queue.ps1 -Queue queue_laptop
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
Set-Location $PSScriptRoot
New-Item -ItemType Directory -Force -Path logs, results | Out-Null
$py = '.\.venv\Scripts\python.exe'
$log = 'logs\queue_laptop.log'
function Log($msg) { Add-Content -Path $log -Value $msg -Encoding UTF8; Write-Output $msg }
# run_app_memory.py reads LongMemEval from a fixed D:\hf_cache path (frozen script), so check the copy first
$need = @('D:\hf_cache\hub\models--Qwen--Qwen3-1.7B',
          'D:\hf_cache\hub\datasets--xiaowu0162--longmemeval-cleaned',
          'runs\q17-dec-s0\final', 'runs\q17-chr-s0\final')
$missing = @($need | Where-Object { -not (Test-Path $_) })
if ($missing.Count -gt 0) { Log "[$((Get-Date).ToString('HH:mm:ss'))] missing (see LAPTOP.md): $($missing -join ', ')"; exit 1 }
$started = Get-Date
$ka = @(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'keepawake\.py' -and $_.CommandLine -notmatch 'Get-CimInstance' })
if ($ka.Count -eq 0) { Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList 'keepawake.py' }
$jobs = New-Object System.Collections.ArrayList
$L = 'Qwen3-1.7B@runs/q17-dec-s0/final,Qwen3-1.7B@runs/q17-chr-s0/final'
# base 1.7B: external evals (probably already complete on the desktop: copy those files over and it is skipped),
# then the application suites
[void]$jobs.Add(@('ext_base_q17',   'run_ext_eval.py --models Qwen3-1.7B'))
[void]$jobs.Add(@('mem_base_q17',   'run_app_memory.py read --models Qwen3-1.7B'))
[void]$jobs.Add(@('logs_base_q17',  'run_app_logs.py --n 100 --models Qwen3-1.7B'))
[void]$jobs.Add(@('agent_base_q17', 'run_app_agent.py --n 150 --models Qwen3-1.7B'))
# seed-0 LoRA pair (queue14 Add-Evals 'q17s0')
[void]$jobs.Add(@('ext_q17s0',   "run_ext_eval.py --models $L"))
[void]$jobs.Add(@('mem_q17s0',   "run_app_memory.py read --models $L"))
[void]$jobs.Add(@('logs_q17s0',  "run_app_logs.py --n 100 --models $L"))
[void]$jobs.Add(@('agent_q17s0', "run_app_agent.py --n 150 --models $L"))
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
if ((Get-Item $PSCommandPath).LastWriteTime -gt $started) { & $PSCommandPath }
# collect everything to bring back on the USB stick
$out = 'laptop_out'
New-Item -ItemType Directory -Force -Path "$out\results", "$out\logs" | Out-Null
Copy-Item results\*Qwen3-1.7B*.jsonl "$out\results\" -Force
Copy-Item logs\* "$out\logs\" -Force
Log "[$((Get-Date).ToString('HH:mm:ss'))] all done; copy the folder $out to the USB stick"
