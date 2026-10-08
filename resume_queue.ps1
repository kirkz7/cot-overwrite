# Continue an experiment queue in the background (after a pause, a crash or a power cut).
# Finished jobs are skipped and each job skips the items it already saved.
# usage: .\resume_queue.ps1                 (exploration queue, queue15.ps1)
#        .\resume_queue.ps1 -Queue queue14  (the large pre-registered queue)
param([string]$Queue = 'queue15')
$running = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'queue\d+\.ps1' -and $_.CommandLine -notmatch 'Get-CimInstance|resume_queue' })
if ($running.Count -gt 0) { "a queue is already running"; exit }
# 10-07 (user approved): start through the Windows Task Scheduler task "cot-<queue>" when it exists, so the queue is
# not a child of the Claude app and survives its service restarts (three queue deaths on 10-05/06/07). The task has no
# trigger; remove it with: Unregister-ScheduledTask cot-queue15
$task = Get-ScheduledTask -TaskName "cot-$Queue" -ErrorAction SilentlyContinue
if ($task) {
    Start-ScheduledTask -TaskName "cot-$Queue"
    "$Queue resumed via scheduled task cot-$Queue; progress: logs\$Queue.log"
} else {
    Start-Process powershell -WindowStyle Hidden -WorkingDirectory $PSScriptRoot -ArgumentList @(
        '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot "$Queue.ps1"))
    "$Queue resumed in the background; progress: logs\$Queue.log"
}
