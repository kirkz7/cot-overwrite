"""E18.2 stop rule (CLOUD_NOTEBOOK "E18.2", written before running): train only if the harvest found at least N looping
thoughts; otherwise exit 1 so the queue stops before training and the result is reported to the user.
usage: python cloud/check_harvest.py data_train/ul_loops8b-e181.jsonl 50"""
import sys

n = sum(1 for _ in open(sys.argv[1], encoding="utf-8"))
print(sys.argv[1], n, "looping thoughts; need", sys.argv[2])
sys.exit(0 if n >= int(sys.argv[2]) else 1)
