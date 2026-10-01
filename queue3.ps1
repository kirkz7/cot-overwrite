# Round 3 GPU queue: Exp 6 (early answering + shuffle on real traces), Exp 7 (mechanism)
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
New-Item -ItemType Directory -Force logs | Out-Null
$py = '.\.venv\Scripts\python.exe'
$jobs = @(
    @('exp6a_early_Qwen3-4B', 'early_answer.py --model Qwen/Qwen3-4B --n 1500'),
    @('exp6b_shuf_Qwen3-4B',  'run_exp6.py --model Qwen/Qwen3-4B --seeds 3'),
    @('exp7_mech_Qwen3-4B',   'mech_exp7.py --model Qwen/Qwen3-4B --n 200')
)
foreach ($j in $jobs) {
    $name, $cmd = $j
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
