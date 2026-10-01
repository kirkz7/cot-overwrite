# Round 4 GPU queue: Exp 8 (long k) and Exp 10 (read/write asymmetry). Runs after queue3.
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
New-Item -ItemType Directory -Force logs | Out-Null
$py = '.\.venv\Scripts\python.exe'
$jobs = @(
    @('exp8_long_Qwen3-4B',      'run_exp8.py --model Qwen/Qwen3-4B --n 150'),
    @('exp8_long_Qwen3-4B-Base', 'run_exp8.py --model Qwen/Qwen3-4B-Base --plain --n 150'),
    @('exp11_monitor_Qwen3-4B',  'run_exp11.py --model Qwen/Qwen3-4B --n 400'),
    @('exp10_rw_Qwen3-4B',       'run_exp10.py --model Qwen/Qwen3-4B --n 100')
)
foreach ($j in $jobs) {
    $name, $cmd = $j
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
