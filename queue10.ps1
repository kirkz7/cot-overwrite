# Non-Qwen scale series on the 5080: OLMo-2 13B (base + instruct), after queue9.
$env:HF_HOME = 'D:\hf_cache'; $env:PYTHONIOENCODING = 'utf-8'; $env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$py = '.\.venv\Scripts\python.exe'
& $py -c "from huggingface_hub import snapshot_download as s; [print(r, s(r, allow_patterns=['*.json','*.safetensors','*.txt','*.model','*.py','*.jinja'])) for r in ['allenai/OLMo-2-1124-13B-Instruct','allenai/OLMo-2-1124-13B']]" *> logs\download_olmo13.log
Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] download done exit=$LASTEXITCODE"
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'queue9\.ps1' }) { Start-Sleep 30 }
$env:HF_HUB_OFFLINE = '1'
Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList 'keepawake.py'
$jobs = New-Object System.Collections.ArrayList
[void]$jobs.Add(@('exp9_e1_OLMo-2-13B',          'run_exp1.py --model allenai/OLMo-2-1124-13B --fmts bare --four_bit --plain'))
[void]$jobs.Add(@('exp9_e2_OLMo-2-13B',          'run_exp2.py --model allenai/OLMo-2-1124-13B --four_bit --plain'))
[void]$jobs.Add(@('exp9_e1_OLMo-2-13B-Instruct', 'run_exp1.py --model allenai/OLMo-2-1124-13B-Instruct --fmts bare --four_bit'))
[void]$jobs.Add(@('exp9_e2_OLMo-2-13B-Instruct', 'run_exp2.py --model allenai/OLMo-2-1124-13B-Instruct --four_bit'))
[void]$jobs.Add(@('exp16_OLMo-2-13B-Instruct',   'run_exp13.py --model allenai/OLMo-2-1124-13B-Instruct --four_bit --n 200'))
foreach ($j in $jobs) {
    $name = $j[0]; $cmd = $j[1]
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
