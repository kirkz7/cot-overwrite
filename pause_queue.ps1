# Stop the running experiment queue and its Python job (e.g. to free the GPU for a game).
# Nothing is lost: finished work is already on disk, and resume_queue.ps1 continues from there.
$procs = @(Get-CimInstance Win32_Process | Where-Object {
    $_.CommandLine -match 'queue(\d*|_laptop)\.ps1|run_app_|run_exp\d|run_ext_eval|train_lora|explore_' -and $_.CommandLine -notmatch 'Get-CimInstance|pause_queue' })
# stop the queue first so it cannot start the next job, then the Python job
$procs | Sort-Object { if ($_.Name -eq 'powershell.exe') { 0 } else { 1 } } | ForEach-Object {
    try { Stop-Process -Id $_.ProcessId -Force -ErrorAction Stop; "stopped $($_.ProcessId) $($_.Name)" } catch {}
}
if ($procs.Count -eq 0) { "no queue was running" } else { "queue paused; GPU memory is free now" }
