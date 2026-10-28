"""Command line: `causalcheck data.csv [--json] [--confounding]`."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

from .report import check


def _load(path: str, sep: str | None) -> pd.DataFrame:
    p = Path(path)
    if not p.exists():
        sys.exit(f"causalcheck: file not found: {path}")
    suffix = p.suffix.lower()
    if suffix in (".parquet", ".pq"):
        return pd.read_parquet(p)
    if suffix in (".xlsx", ".xls"):
        return pd.read_excel(p)
    if suffix in (".tsv", ".tab"):
        return pd.read_csv(p, sep="\t")
    return pd.read_csv(p, sep=sep) if sep else pd.read_csv(p, sep=None, engine="python")


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(
        prog="causalcheck",
        description="Pre-flight assumption checks for causal discovery: which methods your data "
                    "can support, and which outputs to trust.")
    ap.add_argument("file", help="CSV, TSV, Parquet, or Excel file with one column per variable")
    ap.add_argument("--json", action="store_true", help="print a JSON report instead of text")
    ap.add_argument("--confounding", action="store_true",
                    help="also run the hidden-confounder check (needs causal-learn, slower)")
    ap.add_argument("--sep", default=None, help="CSV delimiter (auto-detected if omitted)")
    ap.add_argument("--seed", type=int, default=0, help="random seed for subsampling")
    ap.add_argument("--width", type=int, default=100, help="text report width")
    args = ap.parse_args(argv)

    df = _load(args.file, args.sep)
    report = check(df, confounding=args.confounding, seed=args.seed)
    try:
        print(report.to_json() if args.json else report.to_text(width=args.width))
    except BrokenPipeError:  # e.g. piped into `head`
        pass
    return 1 if report.worst_status == "fail" else 0


if __name__ == "__main__":
    sys.exit(main())
