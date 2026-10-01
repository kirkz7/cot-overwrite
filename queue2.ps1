# Round 2 GPU queue. Each job logs to logs/<name>.log
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
New-Item -ItemType Directory -Force logs | Out-Null
$py = '.\.venv\Scripts\python.exe'
$jobs = @(
    @('exp5_nl_Qwen3-4B',      'run_exp1.py --model Qwen/Qwen3-4B --style nl'),
    @('exp5_nl_Qwen3-4B-Base', 'run_exp1.py --model Qwen/Qwen3-4B-Base --plain --style nl')
)
foreach ($j in $jobs) {
    $name, $cmd = $j
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
