"""Command-line interface for figure reproduction and fresh computations."""

from __future__ import annotations
import argparse
import os
from pathlib import Path
import sys


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Reproduce the PDE controllability-allocation figures."
    )
    sub = parser.add_subparsers(dest="command", required=True)
    figures = sub.add_parser(
        "figures", help="Redraw all four figures using validated saved results."
    )
    figures.add_argument(
        "--data-dir", type=Path, help="A complete adopted or recomputed data bundle."
    )
    figures.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures"),
        help="Output directory for PDF, PNG, and the reproduction report.",
    )
    compute = sub.add_parser(
        "compute",
        help="Recompute FV responses and uniform-start PGD; potentially expensive.",
    )
    compute.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/recomputed"),
        help="New directory for checkpoints and results.",
    )
    compute.add_argument(
        "--resume",
        action="store_true",
        help="Resume only if code, parameters, inputs, and numerical-library versions match.",
    )
    compute.add_argument(
        "--smoke",
        action="store_true",
        help="Run three small cases; NOT a reproduction of all paper results.",
    )
    compute.add_argument(
        "--limit",
        type=int,
        help="Maximum number of new cases in this invocation; use --resume to continue.",
    )
    args = parser.parse_args(argv)
    if args.command == "compute" and args.limit is not None and args.limit < 1:
        parser.error("--limit must be positive")
    # Set these before importing NumPy/SciPy. Do not inherit uncontrolled BLAS parallelism.
    for name in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ[name] = "1"
    os.environ["MPLBACKEND"] = "Agg"
    try:
        if args.command == "figures":
            from .figures import render
            from .dataio import DATA_DIR

            render(args.data_dir or DATA_DIR, args.output)
        else:
            from .compute import run

            result = run(args.output, args.resume, args.smoke, args.limit)
            print(
                "Complete."
                if result["complete"]
                else "Paused; run again with --resume."
            )
    except (ValueError, RuntimeError, FileNotFoundError, FileExistsError) as exc:
        parser.exit(1, f"Error: {exc}\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
