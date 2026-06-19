#!/usr/bin/env python3
"""Extract per-recording QC fields and descriptive biomarkers."""
import argparse, os, time
import pandas as pd
import _bootstrap  # noqa: F401
from src.io_gazebasevr import remap_paths
from src.biomarkers import extract_recording


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw")
    ap.add_argument("--out", default="results")
    a = ap.parse_args()
    man = pd.read_csv(os.path.join(a.out, "vrg_manifest.csv"))
    man = remap_paths(man, a.raw)
    rows, t0 = [], time.time()
    for i, r in man.iterrows():
        try:
            rec = extract_recording(r["path"])
        except Exception as e:
            rec = {"error": str(e)[:40]}
        rec.update(dict(subject=r["subject"], round=r["round"],
                        session=r["session"], file=r["file"]))
        rows.append(rec)
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{len(man)}  ({time.time() - t0:.0f}s)")
    bt = pd.DataFrame(rows)
    path = os.path.join(a.out, "biomarker_table.csv")
    bt.to_csv(path, index=False)
    print("success:", bt["conv_latency_ms"].notna().sum(), "/", len(bt), "->", path)


if __name__ == "__main__":
    main()
