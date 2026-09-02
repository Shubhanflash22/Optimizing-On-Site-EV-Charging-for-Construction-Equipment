#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_live_powers.py
=================================================================================
Generates live_powers.csv (the LIVE_DATA_MODE input) from the raw SOIL
task-recording Excel files, and writes it into every configured MPC/MPC-Shrink
folder.

PIPELINE
  1. Run Tasks_energy_loading_swinging_bayesian.py ONCE, in its existing batch
     mode, to get the fitted posterior mean power for digging / loading+
     swinging / traveling. This is the TARGET MEAN the per-bucket projection
     below is measured against. (Idle's target mean is 0.0 -- see
     IDLE_TARGET_MEAN below for why.)
  2. Re-derive the SAME cumulative-|delta SOC| bucketing used by that
     regression script / 0_Regression.jl, over the 12 soil files only.
  3. For each bucket i, with duration vector A_i (hours per activity) and
     measured energy b_i (kWh, from the SOC drop), solve:
         minimize   ||x - target_mean||^2
         subject to A_i . x = b_i        (exact energy balance for THIS bucket)
                    x >= 0
     giving a real, physically-consistent 4-tuple of activity powers for that
     specific interval -- as close as possible to the established mean, while
     being forced to explain exactly what the telemetry showed.
  4. Record x[a] into activity a's list ONLY when A_i[a] > EPS_HOURS (the
     activity actually occurred in that bucket). If it didn't occur, x[a] is
     unconstrained by the equality and just collapses to target_mean[a] --
     that's not new information, so it is skipped rather than padding the
     list with copies of the mean. This means the 4 activities' lists will
     end up DIFFERENT lengths -- that's expected and fine (live_powers.csv /
     draw_activity_power_pool_live handle ragged per-activity counts already).
  5. Write live_powers.csv (long format: activity,power_kW) into every folder
     in TARGET_DIRS. A folder that doesn't exist is SKIPPED with a printed
     note -- never created, never an error.

EVERYTHING TUNABLE IS IN THE CONFIG BLOCK BELOW.
=================================================================================
"""

import os
import sys
import csv
import subprocess
import numpy as np
import pandas as pd
from scipy.optimize import minimize

# #################################################################################
# CONFIG -- edit anything here without touching the logic below
# #################################################################################

# ---- bucketing ------------------------------------------------------------------
# Cumulative |delta SOC| threshold (%) per bucket/equation. Matches
# MIN_DELTA_SOC in Tasks_energy_loading_swinging_bayesian.py / 0_Regression.jl.
# NOTE: at 3.0, three of the twelve soil files (Oct_22_Tasks_2, Feb_02_Tasks_1,
# Feb_03_Tasks_1) produce ZERO buckets -- their sessions never accumulated a
# 3% cumulative SOC drop. Lower this to recover data from those sessions, at
# the cost of noisier (smaller-drop) buckets.
MIN_DELTA_SOC = 3.0

# Usable battery capacity (kWh) -- converts %SOC drop to kWh consumed.
BATTERY_CAP = 14.8

# Duration (hours) below which an activity is treated as "did not occur" in a
# bucket, and therefore NOT recorded for that activity in that bucket.
EPS_HOURS = 1e-9

# ---- target mean (from the regression script, run once) -------------------------
# Path to the existing Bayesian regression script. It is run as a SEPARATE
# PYTHON PROCESS in its own documented batch mode: passing a destination CSV
# path as argv[1] makes it (a) force the headless Agg matplotlib backend, so
# no blocking plot windows appear, and (b) at the end, write the FULL-DATA
# posterior mean for p_digging / p_loading_swinging / p_traveling (+ their
# per-activity sigma) into that CSV, via its own _export_to_mpc_parameters().
# This is a REAL Bayesian NUTS fit (2000 draws + 2000 tune, 4 chains by
# default in that script) -- expect it to take a few minutes and to require
# the SAME Python environment that already runs it successfully today
# (pymc, pytensor, arviz, xarray, cvxpy, scikit-learn, pandas, matplotlib,
# seaborn). This script does NOT install anything for you.
REGRESSION_SCRIPT = r"C:\Users\shubh\Desktop\Bayesian Regression\Tasks_energy_loading_swinging_bayesian.py"

# Python executable to run REGRESSION_SCRIPT with. Defaults to whatever
# interpreter is running THIS script; override if the regression script needs
# a different environment (e.g. a specific conda env).
REGRESSION_PYTHON = sys.executable

# Where the regression script's batch-mode output gets written (overwritten
# every run of this script).
REGRESSION_OUTPUT_CSV = os.path.join(os.path.dirname(REGRESSION_SCRIPT), "_live_powers_target_mean.csv")

# Set to False to SKIP re-running the (slow) regression fit and instead reuse
# whatever is already sitting at REGRESSION_OUTPUT_CSV from a previous run --
# useful while you're only tuning MIN_DELTA_SOC and don't want to wait for a
# fresh NUTS fit every time.
RUN_REGRESSION = True

# Idle's target mean is 0.0, NOT something the regression script produces --
# idle power is pinned to 0 in that script's own NNLS/Bayesian fit (the
# p_idling >= 0 constraint in its energy-balance program), so there is no
# "fitted idle mean" to read. Using 0.0 as idle's projection anchor keeps the
# SAME convention; the per-bucket QP is still fully free to push idle's value
# away from 0 whenever a bucket's energy balance requires it (that's the
# entire point of generating live_powers.csv in the first place).
IDLE_TARGET_MEAN = 0.0

# ---- soil source files (same 12 files as SOIL_FILES in 0_Regression.jl) ---------
# The .xlsx files are expected to sit next to REGRESSION_SCRIPT (matching that
# script's own _read_excel_anywhere portability shim).
DATA_DIR = os.path.dirname(REGRESSION_SCRIPT)
SOIL_FILES = [
    "Oct_21_Tasks_1.xlsx",
    "Oct_22_Tasks_1.xlsx", "Oct_22_Tasks_2.xlsx", "Oct_22_Tasks_3.xlsx",
    "Oct_22_Tasks_4.xlsx", "Oct_22_Tasks_5.xlsx",
    "Oct_23_Tasks_1.xlsx",
    "Feb_02_Tasks_1.xlsx", "Feb_02_Tasks_2.xlsx", "Feb_02_Tasks_3.xlsx",
    "Feb_03_Tasks_1.xlsx", "Feb_03_Tasks_2.xlsx",
]

# ---- output -----------------------------------------------------------------------
OUTPUT_FILENAME = "live_powers.csv"

# Every folder to (over)write live_powers.csv into. A folder that doesn't
# exist is SKIPPED with a printed note -- never created, never an error.
TARGET_DIRS = [
    r"C:\Users\shubh\Downloads\To be copied\MPC-Shrink\Approach 1\Shrinking_Horizon\data\input_data",
    r"C:\Users\shubh\Downloads\To be copied\MPC-Shrink\Approach 2\Shrinking_Horizon\data\input_data",
    r"C:\Users\shubh\Downloads\To be copied\MPC-Shrink\Comparison_A0_A1_A2\Input",

    r"C:\Users\shubh\Desktop\MPC\Approach 1\Shrinking_Horizon\data\input_data",
    r"C:\Users\shubh\Desktop\MPC\Approach 1\Shrinking_Horizon\data\synthetic_data",
    r"C:\Users\shubh\Desktop\MPC\Approach 1\Receding_Horizon\data\input_data",
    r"C:\Users\shubh\Desktop\MPC\Approach 1\Receding_Horizon\data\synthetic_data",
    r"C:\Users\shubh\Desktop\MPC\Approach 1\Comparison\Input",

    r"C:\Users\shubh\Desktop\MPC\Approach 2\Shrinking_Horizon\data\input_data",
    r"C:\Users\shubh\Desktop\MPC\Approach 2\Shrinking_Horizon\data\synthetic_data",
    r"C:\Users\shubh\Desktop\MPC\Approach 2\Receding_Horizon\data\input_data",
    r"C:\Users\shubh\Desktop\MPC\Approach 2\Receding_Horizon\data\synthetic_data",
    r"C:\Users\shubh\Desktop\MPC\Approach 2\Comparison\Input",

    r"C:\Users\shubh\Desktop\MPC\Comparison_A0_A1_A2\Input",
]

# Activity order matches B = [dig, load+swing, travel, idle] everywhere else
# in the MPC codebase (2_DataLoader.jl's _LIVE_ACTIVITY_NAMES).
ACTIVITY_NAMES = ["p_digging", "p_loading_swinging", "p_traveling", "p_idling"]

# #################################################################################
# STEP 1 -- run the regression script once, read back its fitted mean
# #################################################################################

def get_target_mean() -> np.ndarray:
    """Returns [p_digging, p_loading_swinging, p_traveling, IDLE_TARGET_MEAN]."""
    if RUN_REGRESSION:
        if not os.path.isfile(REGRESSION_SCRIPT):
            raise FileNotFoundError(f"REGRESSION_SCRIPT not found: {REGRESSION_SCRIPT}")
        print(f"[1/3] Running regression script (this fits a real Bayesian NUTS "
              f"model -- expect a few minutes)...\n      {REGRESSION_SCRIPT}")
        result = subprocess.run(
            [REGRESSION_PYTHON, REGRESSION_SCRIPT, REGRESSION_OUTPUT_CSV],
            cwd=os.path.dirname(REGRESSION_SCRIPT),
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"Regression script exited with code {result.returncode}. "
                f"Check that {REGRESSION_PYTHON} has pymc/pytensor/arviz/xarray/"
                f"cvxpy/scikit-learn/pandas/matplotlib/seaborn installed -- the "
                f"same environment that already runs this script successfully "
                f"today. You can also set RUN_REGRESSION = False to reuse a "
                f"previous fit at REGRESSION_OUTPUT_CSV instead of re-fitting."
            )
    else:
        print(f"[1/3] RUN_REGRESSION = False -- reusing existing fit at "
              f"{REGRESSION_OUTPUT_CSV}")

    if not os.path.isfile(REGRESSION_OUTPUT_CSV):
        raise FileNotFoundError(
            f"Expected the regression script's output at {REGRESSION_OUTPUT_CSV} "
            f"but it isn't there. Either set RUN_REGRESSION = True, or point "
            f"REGRESSION_OUTPUT_CSV at an existing fitted-parameters CSV."
        )
    df = pd.read_csv(REGRESSION_OUTPUT_CSV)
    df["Parameter"] = df["Parameter"].astype(str).str.strip()

    def read_param(name):
        row = df.loc[df["Parameter"] == name, "Value"]
        if row.empty:
            raise KeyError(f"'{name}' not found in {REGRESSION_OUTPUT_CSV}")
        return float(row.iloc[0])

    p_dig = read_param("p_digging")
    p_load = read_param("p_loading_swinging")
    p_trav = read_param("p_traveling")
    mean = np.array([p_dig, p_load, p_trav, IDLE_TARGET_MEAN])
    print(f"      target mean (dig, load+swing, travel, idle) = "
          f"{np.round(mean, 4).tolist()} kW")
    return mean

# #################################################################################
# STEP 2 -- bucket each soil file (same algorithm as 0_Regression.jl)
# #################################################################################

def _row_hours(t0, t1) -> float:
    if pd.isna(t0) or pd.isna(t1):
        return 0.0
    try:
        seconds = (t1 - t0).total_seconds()
    except Exception:
        return 0.0
    return seconds / 3600.0 if seconds > 0 else 0.0

# Raw activity label -> which of the 4 model activities it folds into.
# Matches grading == "False" in the regression script / 0_Regression.jl:
# Grading 1 -> digging, Grading 2 -> loading+swinging.
_ACT_FOLD = {
    "Digging": 0, "Grading 1": 0,
    "Loading": 1, "Swinging": 1, "Grading 2": 1,
    "Travelling": 2,
    "Idling": 3,
}

def buckets_from_file(path: str):
    """Returns a list of (A_i: np.ndarray shape (4,), b_i: float) for one file."""
    df = pd.read_excel(path, sheet_name="Sheet1")
    starts = df["Start time (actual)"].tolist()
    stops = df["End time (actual)"].tolist()
    acts = df["Activity"].tolist()
    socs = df["SoC"].tolist()
    n = len(socs)

    durations = [_row_hours(starts[r], stops[r]) for r in range(n)]

    valid_idx = [i for i in range(n) if pd.notna(socs[i])]
    if not valid_idx:
        return []

    buckets = []
    bstart = valid_idx[0]
    anchor = float(socs[bstart])
    j = bstart + 1
    while j < n:
        if pd.isna(socs[j]):
            j += 1
            continue
        soc_now = float(socs[j])
        cum_delta = soc_now - anchor
        if abs(cum_delta) < MIN_DELTA_SOC:
            j += 1
            continue

        h = np.zeros(4)
        for r in range(bstart, j + 1):
            a = acts[r]
            if pd.isna(a):
                continue
            key = str(a).strip()
            if key in _ACT_FOLD:
                h[_ACT_FOLD[key]] += durations[r]
            # unrecognized activity labels (e.g. "Mixing", material-specific)
            # are silently ignored here -- soil files shouldn't have them, but
            # this keeps a mislabeled row from crashing the run.

        b_i = -cum_delta * BATTERY_CAP / 100.0
        buckets.append((h, b_i))

        bstart = j + 1
        anchor = soc_now
        j += 1

    return buckets

# #################################################################################
# STEP 3 -- solve the per-bucket QP
# #################################################################################

def solve_bucket(A_i: np.ndarray, b_i: float, mean: np.ndarray):
    """minimize ||x - mean||^2  s.t.  A_i . x = b_i,  x >= 0.
    Returns x (np.ndarray, shape (4,)), or None if infeasible (only possible
    here if b_i < 0, which does not occur in the soil dataset at 3% -- see
    check_buckets.py -- but is guarded anyway in case the threshold or data
    changes)."""
    if b_i < 0:
        return None

    def obj(x):
        d = x - mean
        return float(d @ d)

    def obj_grad(x):
        return 2.0 * (x - mean)

    cons = [{"type": "eq", "fun": lambda x: A_i @ x - b_i, "jac": lambda x: A_i}]
    bounds = [(0.0, None)] * 4
    x0 = np.clip(mean, 0.0, None)

    res = minimize(obj, x0, jac=obj_grad, bounds=bounds, constraints=cons,
                    method="SLSQP", options={"ftol": 1e-12, "maxiter": 200})
    if not res.success:
        return None
    x = np.clip(res.x, 0.0, None)  # guard against tiny negative numerical noise
    return x

# #################################################################################
# STEP 4 -- assemble + STEP 5 -- write
# #################################################################################

def main():
    mean = get_target_mean()

    print(f"\n[2/3] Bucketing {len(SOIL_FILES)} soil files "
          f"(MIN_DELTA_SOC = {MIN_DELTA_SOC}%)...")
    lists = {name: [] for name in ACTIVITY_NAMES}
    total_buckets = 0
    skipped_infeasible = 0

    for fname in SOIL_FILES:
        path = os.path.join(DATA_DIR, fname)
        if not os.path.isfile(path):
            print(f"      WARNING: soil file missing, skipping: {path}")
            continue
        buckets = buckets_from_file(path)
        n_file = 0
        for A_i, b_i in buckets:
            total_buckets += 1
            x = solve_bucket(A_i, b_i, mean)
            if x is None:
                skipped_infeasible += 1
                print(f"      WARNING: infeasible bucket skipped in {fname} "
                      f"(A={A_i.tolist()}, b={b_i:.4f}) -- see the Infeasible "
                      f"bucket handling note in generate_live_powers.py")
                continue
            for a_idx, name in enumerate(ACTIVITY_NAMES):
                if A_i[a_idx] > EPS_HOURS:
                    lists[name].append(round(float(x[a_idx]), 4))
            n_file += 1
        print(f"      {fname}: {n_file} bucket(s)")

    print(f"      total buckets: {total_buckets} "
          f"({skipped_infeasible} skipped as infeasible)")
    for name in ACTIVITY_NAMES:
        print(f"      {name}: {len(lists[name])} recorded value(s) "
              f"= {lists[name]}")

    print(f"\n[3/3] Writing {OUTPUT_FILENAME} to {len(TARGET_DIRS)} configured folder(s)...")
    written, skipped = 0, 0
    for d in TARGET_DIRS:
        if not os.path.isdir(d):
            print(f"      SKIPPED (folder not found): {d}")
            skipped += 1
            continue
        out_path = os.path.join(d, OUTPUT_FILENAME)
        with open(out_path, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(["activity", "power_kW"])
            for name in ACTIVITY_NAMES:
                for val in lists[name]:
                    w.writerow([name, val])
        print(f"      wrote: {out_path}")
        written += 1

    print(f"\nDone. {written} file(s) written, {skipped} folder(s) skipped "
          f"(not found).")

if __name__ == "__main__":
    main()
