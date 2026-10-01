# Wait for queue3 to finish, then run queue4 and queue5 back to back.
while (Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'queue3\.ps1' }) { Start-Sleep 30 }
Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] queue3 done -> queue4"
& powershell -NoProfile -ExecutionPolicy Bypass -File .\queue4.ps1
Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] queue4 done -> queue5"
& powershell -NoProfile -ExecutionPolicy Bypass -File .\queue5.ps1
Write-Output "[$((Get-Date).ToString('HH:mm:ss'))] all done"
