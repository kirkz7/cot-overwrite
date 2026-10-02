"""Hold a Windows 'system required' power request while the experiment chain is running.

Process-scoped (SetThreadExecutionState): no system setting is changed, and the request is
released as soon as this process exits. Exits by itself once no queue/chain script is running.
"""
import ctypes
import subprocess
import time

ES_CONTINUOUS, ES_SYSTEM_REQUIRED = 0x80000000, 0x00000001


def queue_running():
    # the querying powershell's own command line contains the pattern text, so exclude it
    out = subprocess.run(["powershell", "-NoProfile", "-Command",
                          "@(Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -match 'chain\\.ps1|queue\\d*\\.ps1|run_exp\\d|run_app_|run_ext_eval|train_lora|early_answer\\.py|mech_exp\\d' "
                          "-and $_.CommandLine -notmatch 'Get-CimInstance' }).Count"],
                         capture_output=True, text=True).stdout.strip()
    return out not in ("", "0")


ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS | ES_SYSTEM_REQUIRED)
print(time.strftime("%H:%M:%S"), "keep-awake on", flush=True)
while queue_running():
    time.sleep(60)
ctypes.windll.kernel32.SetThreadExecutionState(ES_CONTINUOUS)
print(time.strftime("%H:%M:%S"), "queues finished; keep-awake released", flush=True)
