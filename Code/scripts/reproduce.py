"""Portable single-thread benchmark wrapper; preserves reference results."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

root = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser()
parser.add_argument("--output", type=Path, default=root / "Results/reproduced")
parser.add_argument("--data-root", type=Path, default=root / "Data/raw/ml-1m")
args = parser.parse_args()
env = os.environ.copy()
for name in ["OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"]:
    env[name] = "1"
subprocess.run([sys.executable, "-m", "matchbox_m2.experiment", "--output", str(args.output),
                "--data-root", str(args.data_root)], env=env, check=True)
subprocess.run([sys.executable, "-m", "matchbox_m2.report_assets", "--results", str(args.output),
                "--latex", str(args.output / "report_assets")], env=env, check=True)
