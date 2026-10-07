"""E18.1 seeds 1-2 (EXPLORE_PLAN "E18.1 多种子"): rebuild the cloud's seed-0 training data on the desktop, exactly.
1. take data/selfdistill_train.jsonl from origin/weights-e181 (pushed by the cloud, commit 7f9f804) -> data_train/selfdistill_train.jsonl
2. python gen_bind_data_v4.py
3. compare the bind4 files with the cloud's hashes, computed on LF line endings (Windows writes CRLF)
Stops with an error if anything differs; then nothing may be trained.
usage: python prep_e181_seeds.py
"""
import hashlib
import subprocess
import sys

SD_SHA = "78a512959c2b2d7aeedada9ea7a3080408dcba8e85961c61cd5a59566b12f5bb"   # data/README.md on weights-e181 (LF)
EXPECTED = {"train": "0B46429FD80CB99A", "val": "12CF83BB5C3139FA", "dev": "CDFADCCFC7031F42"}   # CLOUD_NOTEBOOK, LF


def lf_sha(path):
    return hashlib.sha256(open(path, "rb").read().replace(b"\r\n", b"\n")).hexdigest()[:16].upper()


def main():
    subprocess.run(["git", "fetch", "-q", "origin", "weights-e181"], check=True)
    blob = subprocess.run(["git", "show", "origin/weights-e181:data/selfdistill_train.jsonl"], capture_output=True, check=True).stdout
    open("data_train/selfdistill_train.jsonl", "wb").write(blob)
    sd = hashlib.sha256(blob.replace(b"\r\n", b"\n")).hexdigest()
    print("selfdistill rows", blob.count(b"\n"), "| sha256 (LF)", sd)
    if sd != SD_SHA:
        sys.exit("selfdistill file differs from the cloud's: do not train")
    subprocess.run([sys.executable, "gen_bind_data_v4.py"], check=True)
    ok = True
    for split, want in EXPECTED.items():
        got = lf_sha(f"data_train/bind4_decoupled_{split}.jsonl")
        print(f"bind4 {split}: {got} (cloud {want})", "OK" if got == want else "MISMATCH")
        ok &= got == want
    if not ok:
        sys.exit("training data differs from the cloud's seed 0: do not train")
    print("identical to the cloud's seed-0 data: seeds 1-2 may be trained")


if __name__ == "__main__":
    main()
