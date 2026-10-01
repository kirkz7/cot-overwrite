# Re-run every real-trace experiment after the number-regex fix (numbers ending a sentence
# were skipped). Old outputs are archived in results/old_regex/.
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$py = '.\.venv\Scripts\python.exe'
Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList 'keepawake.py'
$jobs = New-Object System.Collections.ArrayList
[void]$jobs.Add(@('fix_prep_exp4',        'prep_exp4.py'))
[void]$jobs.Add(@('fix_early_Qwen3-4B',   'early_answer.py --model Qwen/Qwen3-4B --n 1500'))
[void]$jobs.Add(@('fix_exp12_Qwen3-4B',   'run_exp12.py --model Qwen/Qwen3-4B --n 400'))
[void]$jobs.Add(@('fix_exp13_Qwen3-4B',   'run_exp13.py --model Qwen/Qwen3-4B --n 400'))
[void]$jobs.Add(@('fix_exp13b_Qwen3-4B',  'run_exp13.py --model Qwen/Qwen3-4B --n 400 --bare'))
[void]$jobs.Add(@('fix_exp11_Qwen3-4B',   'run_exp11.py --model Qwen/Qwen3-4B --n 400'))
[void]$jobs.Add(@('fix_exp6b_Qwen3-4B',   'run_exp6.py --model Qwen/Qwen3-4B --seeds 3'))
[void]$jobs.Add(@('fix_exp13_Qwen3-14B',  'run_exp13.py --model Qwen/Qwen3-14B --four_bit --n 300'))
[void]$jobs.Add(@('fix_exp4_Qwen3-4B',    'run_exp4.py --model Qwen/Qwen3-4B --n_per_group 300'))
foreach ($j in $jobs) {
    $name = $j[0]; $cmd = $j[1]
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
