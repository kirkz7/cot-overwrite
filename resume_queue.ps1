# Continue an experiment queue in the background (after a pause, a crash or a power cut).
# Finished jobs are skipped and each job skips the items it already saved.
# usage: .\resume_queue.ps1                 (exploration queue, queue15.ps1)
#        .\resume_queue.ps1 -Queue queue14  (the large pre-registered queue)
#        .\resume_queue.ps1 -Queue queue_laptop  (on the 8 GB laptop, see LAPTOP.md)
param([string]$Queue = 'queue15')
$running = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'queue(\d+|_laptop)\.ps1' -and $_.CommandLine -notmatch 'Get-CimInstance|resume_queue' })
if ($running.Count -gt 0) { "a queue is already running"; exit }
Start-Process powershell -WindowStyle Hidden -WorkingDirectory $PSScriptRoot -ArgumentList @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot "$Queue.ps1"))
"$Queue resumed in the background; progress: logs\$Queue.log"
