# Sequential GPU queue. Each job logs to logs/<name>.log
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
New-Item -ItemType Directory -Force logs | Out-Null
$py = '.\.venv\Scripts\python.exe'
$jobs = @(
    @('exp4_Qwen3-4B',          'run_exp4.py --model Qwen/Qwen3-4B --n_per_group 300'),
    @('exp1_Qwen3-1.7B',        'run_exp1.py --model Qwen/Qwen3-1.7B'),
    @('exp2_Qwen3-1.7B',        'run_exp2.py --model Qwen/Qwen3-1.7B'),
    @('exp1_Qwen3-4B-Base',     'run_exp1.py --model Qwen/Qwen3-4B-Base --plain'),
    @('exp2_Qwen3-4B-Base',     'run_exp2.py --model Qwen/Qwen3-4B-Base --plain'),
    @('exp1_Phi-4-mini',        'run_exp1.py --model microsoft/Phi-4-mini-instruct'),
    @('exp2_Phi-4-mini',        'run_exp2.py --model microsoft/Phi-4-mini-instruct'),
    @('exp1_Qwen3-8B-4bit',     'run_exp1.py --model Qwen/Qwen3-8B --four_bit'),
    @('exp2_Qwen3-8B-4bit',     'run_exp2.py --model Qwen/Qwen3-8B --four_bit')
)
foreach ($j in $jobs) {
    $name, $cmd = $j
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
