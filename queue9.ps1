# Everything else the 5080 can do before the stage report: Exp 14 (head ablation on real traces),
# Exp 15 (support vs position), Exp 17 (extractor + mitigation), Exp 16 (more readers for Exp 13).
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'queue8\.ps1' }) { Start-Sleep 30 }
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$py = '.\.venv\Scripts\python.exe'
Start-Process -WindowStyle Hidden -FilePath $py -ArgumentList 'keepawake.py'
$jobs = New-Object System.Collections.ArrayList
[void]$jobs.Add(@('exp14_heads_Qwen3-4B',     'run_exp14.py --n 300'))
[void]$jobs.Add(@('exp15_support_Qwen3-4B',   'run_exp15.py --model Qwen/Qwen3-4B --n 300'))
[void]$jobs.Add(@('exp17_extract_Qwen3-4B',   'run_exp17.py --model Qwen/Qwen3-4B --n 300'))
[void]$jobs.Add(@('exp16_Qwen3-8B',           'run_exp13.py --model Qwen/Qwen3-8B --four_bit --n 200'))
[void]$jobs.Add(@('exp16_R1-Distill-7B',      'run_exp13.py --model deepseek-ai/DeepSeek-R1-Distill-Qwen-7B --four_bit --n 200'))
[void]$jobs.Add(@('exp16_OLMo-2-Instruct',    'run_exp13.py --model allenai/OLMo-2-1124-7B-Instruct --four_bit --n 200'))
[void]$jobs.Add(@('exp16_Phi-4-mini',         'run_exp13.py --model microsoft/Phi-4-mini-instruct --n 200'))
[void]$jobs.Add(@('exp15_support_Qwen3-14B',  'run_exp15.py --model Qwen/Qwen3-14B --four_bit --n 200'))
[void]$jobs.Add(@('exp17_extract_Qwen3-14B',  'run_exp17.py --model Qwen/Qwen3-14B --four_bit --n 200'))
foreach ($j in $jobs) {
    $name = $j[0]; $cmd = $j[1]
    $t = Get-Date
    Write-Output "[$($t.ToString('HH:mm:ss'))] start $name"
    & $py $cmd.Split(' ') *> "logs\$name.log"
    Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end   $name exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
}
