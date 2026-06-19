"""Filename parsing, recording manifest, session-local path remapping, loading.

GazeBaseVR vergence filenames follow ``S_<rs>_S<session>_<tasknum>_<task>.csv``
where ``rs`` is a 4-digit code whose first digit is the round and whose last
three digits are the subject id. Only the ``VRG`` task is used here.

Required CSV columns: n (time, ms), lx, rx (per-eye horizontal angle, deg),
xT, zT (target x / depth), clx, cly, clz, crx, cry, crz (eye-centre coords).
"""
from __future__ import annotations
import glob
import re
from pathlib import Path

import pandas as pd

FNAME_PAT = re.compile(
    r"S_(?P<rs>\d{4})_S(?P<session>\d+)_(?P<tasknum>\d+)_(?P<task>[A-Z]+)", re.I
)


def parse_filename(path) -> dict | None:
    """Return {path, file, round, subject, session, task} or None if no match."""
    p = Path(path)
    m = FNAME_PAT.search(p.stem)
    if not m:
        return None
    g = m.groupdict()
    return dict(path=str(p), file=p.name, round=int(g["rs"][0]),
                subject=int(g["rs"][1:]), session=int(g["session"]),
                task=g["task"].upper())


def build_manifest(raw_dir: str, task: str = "VRG") -> pd.DataFrame:
    """Scan ``raw_dir`` recursively and return the manifest for one task."""
    csvs = glob.glob(str(Path(raw_dir) / "**" / "*.csv"), recursive=True)
    rows = [m for m in (parse_filename(p) for p in csvs) if m]
    man = pd.DataFrame(rows)
    if task:
        man = man[man.task == task.upper()].copy()
    return man.reset_index(drop=True)


def index_csvs(raw_dir: str, exclude=("biomarker_table.csv", "vrg_manifest.csv")):
    """Map basename -> absolute path for all raw CSVs in this session.

    Manifest paths go stale between Kaggle sessions; remap with this each run.
    """
    return {Path(p).name: p
            for p in glob.glob(str(Path(raw_dir) / "**" / "*.csv"), recursive=True)
            if Path(p).name not in exclude}


def remap_paths(manifest: pd.DataFrame, raw_dir: str) -> pd.DataFrame:
    """Refresh manifest['path'] against the current session's files."""
    idx = index_csvs(raw_dir)
    m = manifest.copy()
    m["path"] = m["file"].map(idx)
    return m


def find_one(raw_dir: str, name: str):
    """Locate a single helper file (e.g. participant_details.xlsx)."""
    hits = glob.glob(str(Path(raw_dir) / "**" / name), recursive=True)
    return hits[0] if hits else None


def load_recording(path: str) -> pd.DataFrame:
    """Read one recording CSV (raw columns preserved)."""
    return pd.read_csv(path)
