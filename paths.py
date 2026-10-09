"""Data locations (CLOUD_PLAN.md section 4.1). On the desktop nothing is set and every path is the old D:\\ one;
on the cloud machine COT_DATA / HF_HOME point at the big disk (cloud/queue_cloud.sh sets both)."""
import os

DATA_ROOT = os.environ.get("COT_DATA", r"D:\datasets")
HF_HOME = os.environ.get("HF_HOME", r"D:\hf_cache")


def data(*parts):
    """a file under the hand-downloaded data folder (MemConflict, LoCoMo)"""
    return os.path.join(DATA_ROOT, *parts)


def hub(*parts):
    """a path inside the Hugging Face hub cache (snapshot folders; may contain glob patterns)"""
    return os.path.join(HF_HOME, "hub", *parts)


LONGMEMEVAL = hub("datasets--xiaowu0162--longmemeval-cleaned", "snapshots", "98d7416c24c778c2fee6e6f3006e7a073259d48f",
                  "longmemeval_oracle.json")
