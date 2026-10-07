"""Queue guard job: block until no queue runs the given job file (e.g. `cloud/wait_queue.py g12b`), so a two-card job
list does not start while a single-card one still holds a card."""
import subprocess, sys, time
while subprocess.run(["pgrep", "-f", f"queue_cloud.sh cloud/jobs_{sys.argv[1]}.txt"], capture_output=True).returncode == 0:
    time.sleep(30)
