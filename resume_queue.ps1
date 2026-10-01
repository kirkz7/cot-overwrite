# Continue the experiment queue in the background (after a pause, a crash or a power cut).
# Finished jobs are skipped and each job skips the items it already saved.
$running = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'queue14\.ps1' -and $_.CommandLine -notmatch 'Get-CimInstance|resume_queue' })
if ($running.Count -gt 0) { "queue is already running"; exit }
Start-Process powershell -WindowStyle Hidden -WorkingDirectory $PSScriptRoot -ArgumentList @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', (Join-Path $PSScriptRoot 'queue14.ps1'))
"queue resumed in the background; progress: logs\queue14.log"
