# Round 5 GPU queue: Exp 9 post-training lineage + scale. Exp 1 (bare only) and Exp 2 per model.
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
New-Item -ItemType Directory -Force logs | Out-Null
$py = '.\.venv\Scripts\python.exe'
foreach ($job in @(@('exp12_conclusions_Qwen3-4B', 'run_exp12.py --model Qwen/Qwen3-4B --n 400'))) {
    $jn, $cmd = $job
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $jn"
    & $py $cmd.Split(' ') *> "logs\$jn.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $jn exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
$models = @(
    @('allenai/OLMo-2-1124-7B',                  '--four_bit --plain'),
    @('allenai/OLMo-2-1124-7B-SFT',              '--four_bit'),
    @('allenai/OLMo-2-1124-7B-DPO',              '--four_bit'),
    @('allenai/OLMo-2-1124-7B-Instruct',         '--four_bit'),
    @('Qwen/Qwen2.5-Math-7B',                    '--four_bit --plain'),
    @('deepseek-ai/DeepSeek-R1-Distill-Qwen-7B', '--four_bit'),
    @('Qwen/Qwen3-14B',                          '--four_bit')
)
foreach ($m in $models) {
    $name, $flags = $m
    $short = $name.Split('/')[-1]
    foreach ($job in @(@("exp9_e1_$short", "run_exp1.py --model $name --fmts bare $flags"),
                       @("exp9_e2_$short", "run_exp2.py --model $name $flags"))) {
        $jn, $cmd = $job
        $t = Get-Date
        Write-Output "[$($t.ToString('HH:mm:ss'))] start $jn"
        & $py $cmd.Split(' ') *> "logs\$jn.log"
        Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $jn exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
    }
}
