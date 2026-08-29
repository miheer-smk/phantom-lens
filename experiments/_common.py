"""Shared bootstrap for the experiment runners: path setup and argument parsing."""
import argparse, sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

def base_parser(desc):
    ap = argparse.ArgumentParser(description=desc)
    ap.add_argument("--features", help="directory of released feature CSVs (or set PRISM_FEATURES)")
    ap.add_argument("--out", help="output JSON path")
    ap.add_argument("--seed", type=int, default=42)
    return ap

def out_path(a, name):
    return Path(a.out) if a.out else (REPO / "results" / "table_values" / name)
