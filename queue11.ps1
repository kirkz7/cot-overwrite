# Review follow-ups (2026-09-30 night): explicit order cues (18), real-text retraction + symmetric
# control (19), BBH with the model's own CoT (20), read/write redo (10b). Runs after queue10.
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'queue(9|10)\.ps1' }) { Start-Sleep 30 }
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$py = '.\.venv\Scripts\python.exe'
Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList 'keepawake.py'
$jobs = New-Object System.Collections.ArrayList
[void]$jobs.Add(@('exp18_cues_Qwen3-4B',       'run_exp18.py --model Qwen/Qwen3-4B --style sym --n 200'))
[void]$jobs.Add(@('exp18_cues_Qwen3-4B_nl',    'run_exp18.py --model Qwen/Qwen3-4B --style nl --n 200'))
[void]$jobs.Add(@('exp19_retract_Qwen3-4B',    'run_exp19.py --model Qwen/Qwen3-4B --n 300'))
[void]$jobs.Add(@('exp20_bbh_Qwen3-4B',        'run_exp20.py --model Qwen/Qwen3-4B'))
[void]$jobs.Add(@('exp10b_rw_Qwen3-4B',        'run_exp10b.py --model Qwen/Qwen3-4B --n 100'))
[void]$jobs.Add(@('exp18_cues_Qwen3-14B',      'run_exp18.py --model Qwen/Qwen3-14B --four_bit --style sym --n 200'))
[void]$jobs.Add(@('exp19_retract_Qwen3-14B',   'run_exp19.py --model Qwen/Qwen3-14B --four_bit --n 200'))
foreach ($j in $jobs) {
    $name = $j[0]; $cmd = $j[1]
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
