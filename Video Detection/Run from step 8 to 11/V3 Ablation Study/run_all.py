"""
run_all.py
==========
Runs the full pipeline end-to-end in one go:

    1. Step 9  : 3D ResNet training
    2. Step 10 : Final val-only evaluation on the 15% test block + plots

Step 11 (grid search) is intentionally NOT run.

Each child script writes its own clean summary .txt
(step8_summary.txt / step9_summary.txt / step10_summary.txt).

Usage:
    python run_all.py
"""

import sys
import time
import subprocess
from pathlib import Path

# Directory containing this file (and all the step scripts)
SCRIPT_DIR = Path(__file__).resolve().parent

# Ordered list of (label, filename) to execute
PIPELINE = [
    ("Step 9  - ResNet training",     "9.Custom resnet model training.py"),
    ("Step 10 - Final evaluation",    "10.Step 4 - Resnet.py"),
]


def slog(msg=""):
    print(msg)


def run_stage(label, filename):
    script_path = SCRIPT_DIR / filename
    if not script_path.exists():
        slog(f"[SKIP] {label}: file not found -> {script_path}")
        return False

    slog("")
    slog("=" * 70)
    slog(f"RUNNING  {label}")
    slog(f"  file : {script_path.name}")
    slog("=" * 70)

    start = time.time()
    # Stream child output live to this console; -u for unbuffered output.
    result = subprocess.run(
        [sys.executable, "-u", str(script_path)],
        cwd=str(SCRIPT_DIR),
    )
    elapsed = time.time() - start

    ok = (result.returncode == 0)
    status = "OK" if ok else f"FAILED (exit {result.returncode})"
    slog(f"--> {label}: {status}  ({elapsed/60:.1f} min)")
    return ok


def main():
    slog("=" * 70)
    slog("RUN ALL  - Excavator activity pipeline (Steps 9 -> 10)")
    slog("=" * 70)

    overall_start = time.time()

    for label, filename in PIPELINE:
        ok = run_stage(label, filename)
        if not ok:
            slog("")
            slog(f"ABORTING pipeline: '{label}' failed. Later stages not run.")
            break
    else:
        slog("")
        slog("ALL STAGES COMPLETED SUCCESSFULLY.")

    total = time.time() - overall_start
    slog(f"\nTotal wall-clock time: {total/60:.1f} min")


if __name__ == "__main__":
    main()
