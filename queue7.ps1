# Exp 13: copy vs compute dose-response (Qwen3-4B, then Qwen3-14B 4-bit for scale).
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$py = '.\.venv\Scripts\python.exe'
$jobs = New-Object System.Collections.ArrayList
[void]$jobs.Add(@('exp13_Qwen3-4B', 'run_exp13.py --model Qwen/Qwen3-4B --n 400'))
[void]$jobs.Add(@('exp13_Qwen3-14B', 'run_exp13.py --model Qwen/Qwen3-14B --four_bit --n 300'))
foreach ($j in $jobs) {
    $name = $j[0]; $cmd = $j[1]
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
