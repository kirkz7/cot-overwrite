# Wait for chain.ps1 (queue4+5) to finish, then run Exp 12 (it was skipped by a bug in queue5).
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'chain\.ps1' }) { Start-Sleep 30 }
$env:HF_HOME = 'D:\hf_cache'; $env:HF_HUB_OFFLINE = '1'; $env:PYTHONIOENCODING = 'utf-8'
$env:HF_HUB_DISABLE_SYMLINKS_WARNING = '1'
$t = Get-Date
Write-Output "[$($t.ToString('HH:mm:ss'))] start exp12"
& .\.venv\Scripts\python.exe run_exp12.py --model Qwen/Qwen3-4B --n 400 *> "logs\exp12_conclusions_Qwen3-4B.log"
Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] end exp12 exit=$LASTEXITCODE ($([int]((Get-Date) - $t).TotalMinutes) min)"
