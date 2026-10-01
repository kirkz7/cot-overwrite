# Non-Qwen check of the key results: does OLMo-2-13B-Instruct also use explicit order cues
# (Exp 18) and retractions (Exp 19) the way Qwen3-14B does? Plus OLMo-2-7B-Instruct for Exp 18.
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$py = '.\.venv\Scripts\python.exe'
Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList 'keepawake.py'
$jobs = New-Object System.Collections.ArrayList
[void]$jobs.Add(@('exp18_cues_OLMo-2-13B-Instruct',   'run_exp18.py --model allenai/OLMo-2-1124-13B-Instruct --four_bit --style sym --n 200'))
[void]$jobs.Add(@('exp18_cues_OLMo-2-7B-Instruct',    'run_exp18.py --model allenai/OLMo-2-1124-7B-Instruct --four_bit --style sym --n 200'))
[void]$jobs.Add(@('exp19_retract_OLMo-2-13B-Instruct','run_exp19.py --model allenai/OLMo-2-1124-13B-Instruct --four_bit --n 200'))
foreach ($j in $jobs) {
    $name = $j[0]; $cmd = $j[1]
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
