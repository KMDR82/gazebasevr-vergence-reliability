#!/usr/bin/env python3
"""Scan the raw directory and write the VRG recording manifest."""
import argparse, os
import _bootstrap  # noqa: F401
from src.io_gazebasevr import build_manifest


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    man = build_manifest(a.raw, task="VRG")
    os.makedirs(a.out, exist_ok=True)
    path = os.path.join(a.out, "vrg_manifest.csv")
    man.to_csv(path, index=False)
    print(f"VRG recordings: {len(man)} | subjects: {man.subject.nunique()}")
    pairs = man.groupby(["subject", "round"]).session.nunique()
    print(f"(subject,round) with 2 sessions: {int((pairs >= 2).sum())}")
    print("wrote", path)


if __name__ == "__main__":
    main()
