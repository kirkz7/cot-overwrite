# Upgrade-path experiments (2026-09-30 night): Exp 21 (natural competing conclusions in wrong
# R1 traces) and Exp 22 (all 27 BBH tasks: shuffle damage vs state updates). Runs after queue11.
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'queue(9|10|11)\.ps1' }) { Start-Sleep 30 }
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$py = '.\.venv\Scripts\python.exe'
Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList 'keepawake.py'
$jobs = New-Object System.Collections.ArrayList
[void]$jobs.Add(@('exp21_natural_Qwen3-4B', 'run_exp21.py --model Qwen/Qwen3-4B --n 400'))
[void]$jobs.Add(@('exp22_bbh27_Qwen3-4B',   'run_exp22.py --model Qwen/Qwen3-4B'))
foreach ($j in $jobs) {
    $name = $j[0]; $cmd = $j[1]
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
