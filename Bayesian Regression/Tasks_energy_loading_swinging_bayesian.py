#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 15:17:24 2026

@author: avikghosh
"""
"""""""""""""""""""""""""""""""""""""""IMPORT PACKAGES HERE"""""""""""""""""""""""""""""""""

# Auto-relaunch under the CEV_MCS conda env if we're not already in it.
import os, sys
_CEV_PY = "/Users/avikghosh/opt/anaconda3/envs/CEV_MCS/bin/python"
if sys.executable != _CEV_PY and os.path.exists(_CEV_PY):
    os.execv(_CEV_PY, [_CEV_PY, __file__, *sys.argv[1:]])

# Clear any variables from a previous run (only meaningful in Spyder / IPython /
# Jupyter, where the kernel persists; a plain `python file.py` already starts fresh).
try:
    from IPython import get_ipython
    ip = get_ipython()
    if ip is not None:
        ip.run_line_magic("reset", "-f")
except ModuleNotFoundError:
    pass

# Clear terminal (visible screen + scrollback). Works in VS Code's integrated
# terminal, macOS Terminal, iTerm2, and Linux. `os.system("clear")` only clears
# the visible portion in some terminals (notably VS Code), leaving scrollback.
if os.name == "nt":
    os.system("cls")
else:
    print("\033[H\033[2J\033[3J", end="", flush=True)

# Fix PyTensor C compilation on macOS 26 (Tahoe): the SDK's libc++ headers
# need at least C++17. Must be set BEFORE pymc / pytensor is imported.
# If C compilation still fails, replace this line with the Python-only fallback
# below (slower but always works):
os.environ["PYTENSOR_FLAGS"] = "cxx="
#os.environ["PYTENSOR_FLAGS"] = "gcc__cxxflags=-std=c++17 -stdlib=libc++"

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
plt.ion()  # non-blocking plt.show(); script keeps running while figures stay open
import seaborn as sns


import math
import time
from time import process_time
from datetime import datetime, timedelta
import cvxpy as cp
from scipy.optimize import nnls
import re
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split
import statistics
import pymc as pm
import arviz as az
import xarray as xr

start_time = time.time()

"""""""""""""""""""""""""""""""""""""""READ INPUT DATA HERE"""""""""""""""""""""""""""""""""

grading = "False";

# Minimum cumulative |ΔSoC| (in %) per equation. Activities accumulate into a
# bucket until cumulative SoC drop reaches this threshold, then one equation is
# emitted for the full bucket. Set to 1 to reproduce the original behavior of
# one equation per SoC step. Larger values trade equation count for per-equation
# signal-to-noise (1 % steps carry ~50 % relative quantization noise; 2 % steps
# carry ~25 %). Matches Tasks_energy_loading_swinging.py.
MIN_DELTA_SOC = 3

# Material handled. Selects which task-recording files feed the analysis, so you
# don't have to hand-edit all_task_dfs. One of: "soil", "decomposed granite",
# "sand", "all".
#
# NOTE: Data_tasks_1 .. Data_tasks_23 are 23 individual task-recording files
# (Excel sheets), NOT 23 days. They were collected over 9 calendar days
# (Oct 21-23 2025 and Feb 02-13 2026); several files can share the same day.
# MATERIAL_FILES lists the 1-based file numbers for each material.
MATERIAL = "soil"

MATERIAL_FILES = {
    "soil":               list(range(1, 13)),   # files 1-12  (Oct 21-23 2025, Feb 02-03 2026)
    "decomposed_granite": list(range(13, 18)),  # files 13-17 (Feb 04 + Feb 11 2026, Site 1)
    "sand":               list(range(18, 24)),  # files 18-23 (Feb 11-13 2026, Site 2)
}
MATERIAL_FILES["all"] = list(range(1, 24))       # every file

if MATERIAL not in MATERIAL_FILES:
    raise ValueError(f"MATERIAL must be one of {sorted(MATERIAL_FILES)}, got {MATERIAL!r}")

"""""""""""""""""""""""""""""""""""""""October 2025: Site 1: Soil"""""""""""""""""""""""""""""""""

################## October 21 2025: Site 1: Soil

Data_tasks_1 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Oct_21_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

################## October 22 2025: Site 1: Soil

Data_tasks_2 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Oct_22_Tasks_1.xlsx',
    sheet_name="Sheet1"
)
Data_tasks_3 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Oct_22_Tasks_2.xlsx',
    sheet_name="Sheet1"
)
Data_tasks_4 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Oct_22_Tasks_3.xlsx',
    sheet_name="Sheet1"
)
Data_tasks_5 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Oct_22_Tasks_4.xlsx',
    sheet_name="Sheet1"
)
Data_tasks_6 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Oct_22_Tasks_5.xlsx',
    sheet_name="Sheet1"
)

################## October 23 2025: Site 1: Soil

Data_tasks_7 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Oct_23_Tasks_1.xlsx',
    sheet_name="Sheet1"
)


"""""""""""""""""""""""""""""""""""""""February 2026: Site 1: Soil"""""""""""""""""""""""""""""""""

################## February 02 2026: Site 1: Soil

Data_tasks_8 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_02_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_9 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_02_Tasks_2.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_10 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_02_Tasks_3.xlsx',
    sheet_name="Sheet1"
)



################## February 03 2026: Site 1: Soil


Data_tasks_11 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_03_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_12 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_03_Tasks_2.xlsx',
    sheet_name="Sheet1"
)


################## February 04 2026: Site 1: Decomposed Granite


Data_tasks_13 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_04_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_14 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_04_Tasks_2.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_15 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_04_Tasks_3.xlsx',
    sheet_name="Sheet1"
)

################## February 11 2026: Site 1: Decomposed Granite

Data_tasks_16 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_11_Tasks_2.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_17 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_11_Tasks_3.xlsx',
    sheet_name="Sheet1"
)

################## February 11 2026: Site 2: Sand


Data_tasks_18 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_11_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

################## February 12 2026: Site 2: Sand


Data_tasks_19 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_12_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_20 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_12_Tasks_2.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_21 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_12_Tasks_3.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_22 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_12_Tasks_4.xlsx',
    sheet_name="Sheet1"
)

################## February 13 2026: Site 2: Sand

Data_tasks_23 = pd.read_excel(
    r'/Users/avikghosh/Desktop/CEV-Analysis/Analysis/Feb_13_Tasks_1.xlsx',
    sheet_name="Sheet1"
)



Battery_cap = 14.8

"""""""""""""""""""""""""""""""""""""""HELPER FUNCTIONS"""""""""""""""""""""""""""""""""

def prepare_task_data(df):
    """
    Keep needed columns, parse timestamps, compute duration in seconds,
    and remove negative durations.
    """
    df_clean = df[['Start time (actual)', 'End time (actual)', 'Activity', 'SoC']].copy()

    df_clean["Start time (actual)"] = pd.to_datetime(df_clean["Start time (actual)"], errors="coerce")
    df_clean["End time (actual)"]   = pd.to_datetime(df_clean["End time (actual)"], errors="coerce")

    df_clean["duration_s"] = (
        df_clean["End time (actual)"] - df_clean["Start time (actual)"]
    ).dt.total_seconds()

    df_clean.loc[df_clean["duration_s"] < 0, "duration_s"] = np.nan

    return df_clean


def build_equations_from_tasks(df_clean, battery_cap, grading):
    """
    Walk rows accumulating activity time into a bucket. Emit one equation when
    cumulative |ΔSoC| from the bucket's anchor row reaches MIN_DELTA_SOC, then
    start a fresh bucket anchored at the current row.

    Cumulative-bucket attribution preserves total energy when MIN_DELTA_SOC > 1:
    1% drops that don't individually clear the threshold are merged into the
    next equation along with their activity time, so no SoC drop is discarded.

    Setting MIN_DELTA_SOC = 1 reproduces one-equation-per-step.

    grading == "False": 5 columns
        [Digging+Grading1, Loading+Swinging+Grading2, Travelling, Idling, Mixing]
    grading == "True":  7 columns
        [Digging, Grading1, Loading+Swinging, Grading2, Travelling, Idling, Mixing]
    """
    A_rows_all = []
    b_rows_all = []

    if len(df_clean) == 0:
        return A_rows_all, b_rows_all

    # Anchor the first bucket at the first valid SoC reading.
    bucket_start = 0
    while bucket_start < len(df_clean) and pd.isna(df_clean.iloc[bucket_start]["SoC"]):
        bucket_start += 1
    if bucket_start >= len(df_clean):
        return A_rows_all, b_rows_all
    bucket_anchor_soc = df_clean.iloc[bucket_start]["SoC"]

    for j in range(bucket_start + 1, len(df_clean)):
        soc_now = df_clean.iloc[j]["SoC"]
        if pd.isna(soc_now):
            continue

        cumulative_delta = soc_now - bucket_anchor_soc
        if abs(cumulative_delta) < MIN_DELTA_SOC:
            continue

        df_slice = df_clean.iloc[bucket_start:j + 1]

        total_s_Digging    = df_slice.loc[df_slice["Activity"] == "Digging",    "duration_s"].sum()
        total_s_Grading_1  = df_slice.loc[df_slice["Activity"] == "Grading 1",  "duration_s"].sum()
        total_s_Loading    = df_slice.loc[df_slice["Activity"] == "Loading",    "duration_s"].sum()
        total_s_Swinging   = df_slice.loc[df_slice["Activity"] == "Swinging",   "duration_s"].sum()
        total_s_Grading_2  = df_slice.loc[df_slice["Activity"] == "Grading 2",  "duration_s"].sum()
        total_s_Travelling = df_slice.loc[df_slice["Activity"] == "Travelling", "duration_s"].sum()
        total_s_Idling     = df_slice.loc[df_slice["Activity"] == "Idling",     "duration_s"].sum()
        total_s_Mixing     = df_slice.loc[df_slice["Activity"] == "Mixing",     "duration_s"].sum()

        total_energy = -cumulative_delta * battery_cap / 100

        if grading == "False":
            A_row = [
                total_s_Digging/3600 + total_s_Grading_1/3600,
                total_s_Loading/3600 + total_s_Swinging/3600 + total_s_Grading_2/3600,
                total_s_Travelling/3600,
                total_s_Idling/3600,
                total_s_Mixing/3600,
            ]
        else:
            A_row = [
                total_s_Digging/3600,
                total_s_Grading_1/3600,
                total_s_Loading/3600 + total_s_Swinging/3600,
                total_s_Grading_2/3600,
                total_s_Travelling/3600,
                total_s_Idling/3600,
                total_s_Mixing/3600,
            ]

        A_rows_all.append(A_row)
        b_rows_all.append([total_energy])

        bucket_start = j + 1
        bucket_anchor_soc = soc_now

    return A_rows_all, b_rows_all


def split_train_test(A, b, test_size, random_state=42, shuffle=True):
    A_train, A_test, b_train, b_test = train_test_split(
        A, b,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
    return A_train, A_test, b_train, b_test


def gamma_from_mean_std(mean, std):
    """
    Convert desired mean and std to Gamma(alpha, beta) parameters.
    PyMC parameterization:
        mean = alpha / beta
        var  = alpha / beta^2
    """
    mean = np.asarray(mean, dtype=float)
    std = np.asarray(std, dtype=float)
    var = std ** 2
    alpha = mean ** 2 / var
    beta = mean / var
    return alpha, beta


def build_prior_x(name, prior_config, dims):
    """
    Supported x priors:
    - halfnormal
    - truncated_normal
    - lognormal
    - gamma
    - exponential
    """
    dist = prior_config["dist"].lower()

    if dist == "halfnormal":
        sigma = np.asarray(prior_config["sigma"], dtype=float)
        return pm.HalfNormal(name, sigma=sigma, dims=dims)

    elif dist == "truncated_normal":
        mu = np.asarray(prior_config["mu"], dtype=float)
        sigma = np.asarray(prior_config["sigma"], dtype=float)
        lower = float(prior_config.get("lower", 0.0))
        upper = prior_config.get("upper", None)
        return pm.TruncatedNormal(name, mu=mu, sigma=sigma, lower=lower, upper=upper, dims=dims)

    elif dist == "lognormal":
        mu = np.asarray(prior_config["mu"], dtype=float)
        sigma = np.asarray(prior_config["sigma"], dtype=float)
        return pm.LogNormal(name, mu=mu, sigma=sigma, dims=dims)

    elif dist == "gamma":
        alpha = np.asarray(prior_config["alpha"], dtype=float)
        beta = np.asarray(prior_config["beta"], dtype=float)
        return pm.Gamma(name, alpha=alpha, beta=beta, dims=dims)

    elif dist == "exponential":
        lam = np.asarray(prior_config["lam"], dtype=float)
        return pm.Exponential(name, lam=lam, dims=dims)

    else:
        raise ValueError(f"Unsupported x prior distribution: {prior_config['dist']}")


def build_prior_sigma(name, prior_config):
    """
    Supported sigma priors:
    - halfnormal
    - exponential
    - halfstudentt
    """
    dist = prior_config["dist"].lower()

    if dist == "halfnormal":
        sigma = float(prior_config["sigma"])
        return pm.HalfNormal(name, sigma=sigma)

    elif dist == "exponential":
        lam = float(prior_config["lam"])
        return pm.Exponential(name, lam=lam)

    elif dist == "halfstudentt":
        sigma = float(prior_config["sigma"])
        nu = float(prior_config["nu"])
        return pm.HalfStudentT(name, sigma=sigma, nu=nu)

    else:
        raise ValueError(f"Unsupported sigma prior distribution: {prior_config['dist']}")


def fit_bayesian_activity_power(grading,
    A,
    b,
    x_prior_config,
    sigma_prior_config,
    draws=2000,
    tune=2000,
    random_seed=42,
    target_accept=0.9
):
    """
    Flexible Bayesian regression model:
        b_i | x, sigma ~ Normal(A_i x, sigma)
    """

    A = np.asarray(A, dtype=float)
    b = np.asarray(b, dtype=float).flatten()

    N, m = A.shape

    if b.shape[0] != N:
        raise ValueError(f"b must have length {N}, but got {b.shape[0]}")

    if (grading == "False"):
        
        activity_names = ["Digging", "Loading + Swinging", "Traveling", "Idling", "Mixing"]

        if len(activity_names) != m:
            raise ValueError(f"Expected {len(activity_names)} activities, but A has {m} columns")

        with pm.Model(coords={"activity": activity_names}) as model:
            x = build_prior_x("x", x_prior_config, dims="activity")
            sigma = build_prior_sigma("sigma", sigma_prior_config)

            mu = pm.math.dot(A, x)
            pm.Normal("b_obs", mu=mu, sigma=sigma, observed=b)

            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=4,
                cores=4,
                random_seed=random_seed,
                target_accept=target_accept,
                return_inferencedata=True,
                nuts_sampler="nutpie",  # Rust-based sampler: faster, no Python multiprocessing
            )
            
    else:
        
        activity_names = ["Digging", "Grading 1", "Loading + Swinging", "Grading 2", "Traveling", "Idling", "Mixing"]

        if len(activity_names) != m:
            raise ValueError(f"Expected {len(activity_names)} activities, but A has {m} columns")

        with pm.Model(coords={"activity": activity_names}) as model:
            x = build_prior_x("x", x_prior_config, dims="activity")
            sigma = build_prior_sigma("sigma", sigma_prior_config)

            mu = pm.math.dot(A, x)
            pm.Normal("b_obs", mu=mu, sigma=sigma, observed=b)

            trace = pm.sample(
                draws=draws,
                tune=tune,
                chains=4,
                cores=4,
                random_seed=random_seed,
                target_accept=target_accept,
                return_inferencedata=True,
                nuts_sampler="nutpie",  # Rust-based sampler: faster, no Python multiprocessing
            )

    return model, trace


def predictive_interval_coverage_from_trace(A_test, b_test, trace, alpha=0.05, random_seed=42):
    """
    Compute posterior predictive mean, predictive intervals, and coverage.
    """
    rng = np.random.default_rng(random_seed)

    A_test = np.asarray(A_test, dtype=float)
    b_test = np.asarray(b_test, dtype=float).flatten()

    x_samples = trace.posterior["x"].stack(sample=("chain", "draw")).values.T
    sigma_samples = trace.posterior["sigma"].stack(sample=("chain", "draw")).values.flatten()

    N_test = A_test.shape[0]
    S = x_samples.shape[0]

    mu_samples = A_test @ x_samples.T
    pred_mean = np.mean(mu_samples, axis=1)

    y_pred_samples = mu_samples + rng.normal(
        loc=0.0,
        scale=sigma_samples,
        size=(N_test, S)
    )

    lower = np.quantile(y_pred_samples, alpha / 2, axis=1)
    upper = np.quantile(y_pred_samples, 1 - alpha / 2, axis=1)

    inside = (b_test >= lower) & (b_test <= upper)
    coverage = np.mean(inside)

    rmse = np.sqrt(np.mean((pred_mean - b_test) ** 2))
    mae = np.mean(np.abs(pred_mean - b_test))
    avg_width = np.mean(upper - lower)

    return pred_mean, lower, upper, inside, coverage, rmse, mae, avg_width


def print_posterior_means(trace, col_present):
    x_post_mean = trace.posterior["x"].mean(dim=("chain", "draw")).values
    sigma_post_mean = trace.posterior["sigma"].mean(dim=("chain", "draw")).values.item()

    if (grading == "False"):
        labels = ["Digging", "Loading+Swinging", "Traveling", "Idling", "Mixing"]
    else:
        labels = ["Digging", "Grading 1", "Loading+Swinging", "Grading 2",
                  "Traveling", "Idling", "Mixing"]

    # Skip activities that never occur in the dataset.
    for i, lab in enumerate(labels):
        if col_present[i]:
            print(f"{lab + ':':20s}{x_post_mean[i]:.3f} kW")
    print(f"{'Sigma:':20s}{sigma_post_mean:.3f}")


def label_posterior_axes(axes, ds, var_names):
    """Set the x-label and rewrite az.plot_posterior's built-in 'mean=...' label
    in place to 'mean ± SD = <m> ± <s> kW', so the mean and SD share the built-in
    location/format. Axes (row-major) are matched to var_names in order."""
    flat = [ax for ax in np.ravel(axes) if ax is not None]
    for ax, name in zip(flat, var_names):
        vals = np.asarray(ds[name].values)
        m, s = vals.mean(), vals.std()
        ax.set_xlabel("Power (kW)")
        new_text = f"mean ± SD = {m:.2f} ± {s:.2f} kW"
        for t in ax.texts:
            if t.get_text().lower().startswith("mean"):
                t.set_text(new_text)
                break
        else:  # fallback if the built-in label wasn't found
            ax.text(0.5, 0.9, new_text, transform=ax.transAxes,
                    ha="center", va="top", fontsize=8)


"""""""""""""""""""""""""""""""""""""""PREPARE DATA"""""""""""""""""""""""""""""""""

Data_tasks_clean_1 = prepare_task_data(Data_tasks_1)        # Oct 21, 2025: Soil

Data_tasks_clean_2 = prepare_task_data(Data_tasks_2)
Data_tasks_clean_3 = prepare_task_data(Data_tasks_3)
Data_tasks_clean_4 = prepare_task_data(Data_tasks_4)
Data_tasks_clean_5 = prepare_task_data(Data_tasks_5)   
Data_tasks_clean_6 = prepare_task_data(Data_tasks_6)   
Data_tasks_clean_7 = prepare_task_data(Data_tasks_7)        # Oct 22-23, 2025: Soil


Data_tasks_clean_8 = prepare_task_data(Data_tasks_8)   
Data_tasks_clean_9 = prepare_task_data(Data_tasks_9)   
Data_tasks_clean_10 = prepare_task_data(Data_tasks_10)      # Feb 02, 2026: Soil

Data_tasks_clean_11 = prepare_task_data(Data_tasks_11)      
Data_tasks_clean_12 = prepare_task_data(Data_tasks_12)      # Feb 03, 2026: Soil

Data_tasks_clean_13 = prepare_task_data(Data_tasks_13)
Data_tasks_clean_14 = prepare_task_data(Data_tasks_14)
Data_tasks_clean_15 = prepare_task_data(Data_tasks_15)      # Feb 04, 2026: Decomposed Granite

Data_tasks_clean_16 = prepare_task_data(Data_tasks_16)
Data_tasks_clean_17 = prepare_task_data(Data_tasks_17)      # Feb 11, 2026: Decomposed Granite

Data_tasks_clean_18 = prepare_task_data(Data_tasks_18)      # Feb 11, 2026: Sand

Data_tasks_clean_19 = prepare_task_data(Data_tasks_19)
Data_tasks_clean_20 = prepare_task_data(Data_tasks_20)   
Data_tasks_clean_21 = prepare_task_data(Data_tasks_21)   
Data_tasks_clean_22 = prepare_task_data(Data_tasks_22)      # Feb 12, 2026: Sand

Data_tasks_clean_23 = prepare_task_data(Data_tasks_23)      # Feb 13, 2026: Sand



Data_tasks_clean = [Data_tasks_clean_1,Data_tasks_clean_2, Data_tasks_clean_3, Data_tasks_clean_4, Data_tasks_clean_5, Data_tasks_clean_6, 
                    Data_tasks_clean_7,Data_tasks_clean_8, Data_tasks_clean_9, Data_tasks_clean_10, Data_tasks_clean_11, Data_tasks_clean_12, 
                    Data_tasks_clean_13,Data_tasks_clean_14, Data_tasks_clean_15, Data_tasks_clean_16, Data_tasks_clean_17, Data_tasks_clean_18, 
                    Data_tasks_clean_19,Data_tasks_clean_20, Data_tasks_clean_21, Data_tasks_clean_22, Data_tasks_clean_23 
                    ]

Data_tasks_clean_combined = pd.concat(Data_tasks_clean, ignore_index=True)


"""""""""""""""""""""""""""""""""""""""FORM EQUATIONS FROM SELECTED TASK FILES"""""""""""""""""""""""""""""""""

# Automatically pick the task files for the selected MATERIAL. Data_tasks_clean
# holds all 23 task files in order, so file f (1-based) is Data_tasks_clean[f - 1].
selected_files = MATERIAL_FILES[MATERIAL]
all_task_dfs = [Data_tasks_clean[f - 1] for f in selected_files]
print(f"MATERIAL = {MATERIAL!r}  ->  using task files {selected_files} ({len(all_task_dfs)} files)")

A = []
b = []

for df_clean in all_task_dfs:
    A_part, b_part = build_equations_from_tasks(df_clean, Battery_cap, grading)
    A.extend(A_part)
    b.extend(b_part)

df = pd.concat(all_task_dfs)
unique_tasks = df["Activity"].unique()

print(f"\n\n************OUTPUT************\n\n")

print(f"The number of unique tasks are in the dataset are: {unique_tasks}")

"""""""""""""""""""""""""""""""""""""""DEFINING MATRIX AND VECTOR"""""""""""""""""""""""""""""""""

A = np.asarray(np.array(A), dtype=float)
b = np.asarray(np.array(b), dtype=float).reshape(-1)

m, n = A.shape

# Which activities actually occur in the dataset. A column that is all-zero
# (e.g. Mixing on soil-only days) carries no data, so its coefficient is driven
# entirely by the prior — we suppress those activities from the printout/plots.
col_present = A.sum(axis=0) > 1e-12

# Time_Digging = np.sum(A[:, [0]], axis=0)
# Time_Loading_Swinging = np.sum(A[:, [1]], axis=0)
# Time_Traveling = np.sum(A[:, [2]], axis=0)

# Time_all = Time_Digging + Time_Loading_Swinging + Time_Traveling

"""""""""""""""""""""""""""""""""""""""SPLIT TRAINING AND TESTING"""""""""""""""""""""""""""""""""

A_train, A_test, b_train, b_test = split_train_test(A, b, test_size=0.2)

"""""""""""""""""""""""""""""""""""""""CHOOSE PRIOR SETUP HERE"""""""""""""""""""""""""""""""""

# ---------- Option 1: HalfNormal prior on x ----------

# if (grading == "False"):
#     x_prior_config = {
#         "dist": "halfnormal",
#         "sigma": np.array([5.7, 4.0, 5.8, 0.000001])
#     }
# else:
#     x_prior_config = {
#         "dist": "halfnormal",
#         "sigma": np.array([6, 5, 4, 4, 5.9, 0.00001])
#     }
# ---------- Option 2: TruncatedNormal prior on x (per material) ----------
# ENTER THE PRIORS HERE. Each activity power uses TruncatedNormal(mu, sigma,
# lower=0). Values are looked up by MATERIAL (and by grading, since the number
# of activities differs). Array order MUST match the activity columns:
#   grading == "False": [Digging, Loading+Swinging, Traveling, Idling, Mixing]
#   grading == "True" : [Digging, Grading 1, Loading+Swinging, Grading 2,
#                        Traveling, Idling, Mixing]
# Idling is kept pinned near 0 (mu=0, small sigma). Where an activity is absent
# for a material (e.g. Mixing on soil), the data can't identify it, so its prior
# value doesn't affect predictions.
X_PRIOR_MU_SIGMA = {
    "False": {   # grading == "False"   (5 activities)
        #                       Digging  Load+Swing  Travel  Idling  Mixing
        "soil":               {"mu":    [4.79, 3.16, 4.71, 0.00, 0.00],
                               "sigma": [0.23, 0.23, 0.54, 0.10, 0.10]},
        "decomposed_granite": {"mu":    [3.12, 3.25, 7.29, 0.00, 0.00],
                               "sigma": [1.42, 0.52, 0.91, 0.10, 0.10]},
        "sand":               {"mu":    [9.12, 4.56, 0.15, 0.00, 5.04],
                               "sigma": [4.98, 0.87, 1.14, 0.10, 0.21]},
        "all":                {"mu":    [4.60, 3.00, 4.50, 0.00, 4.70],
                               "sigma": [1.00, 1.00, 1.50, 0.10, 1.00]},
    },
    "True": {    # grading == "True"    (7 activities)
        #                       Digging  Grading1  Load+Swing  Grading2  Travel  Idling  Mixing
        "soil":               {"mu":    [0.00, 5.50, 3.50, 5.40, 1.00, 0.00, 4.70],
                               "sigma": [0.10, 1.00, 1.00, 1.00, 1.50, 0.10, 0.10]},
        "decomposed_granite": {"mu":    [0.00, 5.50, 3.50, 5.40, 1.00, 0.00, 4.70],
                               "sigma": [0.10, 1.00, 1.00, 1.00, 1.50, 0.10, 1.00]},
        "sand":               {"mu":    [0.00, 5.50, 3.50, 5.40, 1.00, 0.00, 4.70],
                               "sigma": [0.10, 1.00, 1.00, 1.00, 1.50, 0.10, 1.00]},
        "all":                {"mu":    [0.00, 5.50, 3.50, 5.40, 1.00, 0.00, 4.70],
                               "sigma": [0.10, 1.00, 1.00, 1.00, 1.50, 0.10, 1.00]},
    },
}

if MATERIAL not in X_PRIOR_MU_SIGMA[grading]:
    raise ValueError(f"No x-prior defined for MATERIAL={MATERIAL!r} under grading={grading!r}")

_mp = X_PRIOR_MU_SIGMA[grading][MATERIAL]
x_prior_config = {
    "dist": "truncated_normal",
    "mu": np.array(_mp["mu"], dtype=float),
    "sigma": np.array(_mp["sigma"], dtype=float),
    "lower": 0.0,
}
print(f"x prior (material={MATERIAL!r}, grading={grading!r}): "
      f"mu={x_prior_config['mu'].tolist()}, sigma={x_prior_config['sigma'].tolist()}")

# ---------- Option 3: LogNormal prior on x ----------
# x_prior_config = {
#     "dist": "lognormal",
#     "mu": np.log(np.array([5.9, 4.0, 6.5, 1.0])),
#     "sigma": np.array([0.35, 0.35, 0.35, 0.50])
# }

# ---------- Option 4: Gamma prior on x ----------
# prior_mean = np.array([5.9, 4.0, 6.5, 1.0])
# prior_std  = np.array([2.0, 1.5, 2.5, 0.5])
# alpha_x, beta_x = gamma_from_mean_std(prior_mean, prior_std)
# x_prior_config = {
#     "dist": "gamma",
#     "alpha": alpha_x,
#     "beta": beta_x
# }

# Build the sigma prior from whichever b vector the fit actually uses, so the
# scale matches the data being fit: the train/test fit below uses b_train, while
# the full-data fit (Option A) uses the full b.
def make_sigma_prior_config(b_vec):
    return {
        "dist": "halfnormal",
        "sigma": float(np.std(b_vec))
    }
    # Alternatives:
    # return {"dist": "exponential", "lam": 1.0 / max(float(np.std(b_vec)), 1e-6)}
    # return {"dist": "halfstudentt", "sigma": float(np.std(b_vec)), "nu": 4}

sigma_prior_config = make_sigma_prior_config(b_train)

"""""""""""""""""""""""""""""""""""""""BAYESIAN REGRESSION"""""""""""""""""""""""""""""""""

model, trace = fit_bayesian_activity_power(grading,
    A_train,
    b_train,
    x_prior_config=x_prior_config,
    sigma_prior_config=sigma_prior_config,
    draws=2000,
    tune=2000,
    random_seed=42,
    target_accept=0.9
)

"""""""""""""""""""""""""""""""""""""""PLOTTING"""""""""""""""""""""""""""""""""

print_posterior_means(trace, col_present)

# az.plot_posterior(trace, var_names=["x", "sigma"], hdi_prob=0.95)

# (display label, model 'activity' coordinate) in column order.
if (grading == "False"):
    label_to_coord = [
        ("Digging", "Digging"),
        ("Loading+Swinging", "Loading + Swinging"),
        ("Traveling", "Traveling"),
        ("Idling", "Idling"),
        ("Mixing", "Mixing"),
    ]
else:
    label_to_coord = [
        ("Digging", "Digging"),
        ("Grading 1", "Grading 1"),
        ("Loading+Swinging", "Loading + Swinging"),
        ("Grading 2", "Grading 2"),
        ("Traveling", "Traveling"),
        ("Idling", "Idling"),
        ("Mixing", "Mixing"),
    ]

# Keep only activities that occur in the dataset (drop all-zero columns).
present = [lc for i, lc in enumerate(label_to_coord) if col_present[i]]

posterior_renamed = xr.Dataset({
    lab: trace.posterior["x"].sel(activity=coord).reset_coords(drop=True)
    for lab, coord in present
})
var_names = [lab for lab, _ in present]

ncols = min(3, len(var_names))
nrows = int(np.ceil(len(var_names) / ncols))

axes = az.plot_posterior(
    posterior_renamed,
    var_names=var_names,
    hdi_prob=0.95,
    grid=(nrows, ncols),
    figsize=(4 * ncols, 3.5 * nrows)
)

label_posterior_axes(axes, posterior_renamed, var_names)

plt.tight_layout()
plt.show()
"""""""""""""""""""""""""""""""""""""""POSTERIOR PREDICTIVE VALIDATION"""""""""""""""""""""""""""""""""

pred_mean, lower, upper, inside, coverage, rmse, mae, avg_width = \
    predictive_interval_coverage_from_trace(A_test, b_test, trace, alpha=0.05, random_seed=42)

# MAE as a percentage of the observed b_test.
# MAPE: average of |b - b_pred| / |b| (per-sample relative error)
mape = np.mean(np.abs((b_test - pred_mean) / b_test)) * 100
# Normalized MAE: MAE divided by mean(|b_test|). More stable when b_test has
# values close to zero (avoids exploding per-sample ratios).
nmae = mae / np.mean(np.abs(b_test)) * 100

print("\n---------------- TEST RESULTS ----------------")
print(f"Test RMSE:          {rmse:.4f} kWh")
print(f"Test MAE:           {mae:.4f} kWh")
print(f"Test MAPE:          {mape:.2f} %   (mean |b - b_pred| / |b|)")
print(f"Test MAE / mean|b|: {nmae:.2f} %   (MAE relative to average |b|)")
print(f"95% predictive interval coverage: {coverage:.4f}")
print(f"Average predictive interval width: {avg_width:.4f} kWh")

results_df = pd.DataFrame({
    "Observed_b_test": b_test,
    "Predicted_mean": pred_mean,
    "PI_lower_95": lower,
    "PI_upper_95": upper,
    "Inside_95_PI": inside
})

print("\nTest prediction summary:")
print(results_df)

plt.figure(figsize=(6, 6))
plt.scatter(b_test, pred_mean)
min_val = min(np.min(b_test), np.min(pred_mean))
max_val = max(np.max(b_test), np.max(pred_mean))
plt.plot([min_val, max_val], [min_val, max_val], '--')
plt.xlabel("Observed test energy (kWh)")
plt.ylabel("Predicted mean energy (kWh)")
plt.title("Observed vs Predicted on Test Set")
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
idx = np.arange(1, len(b_test) + 1)

plt.plot(idx, b_test, 'o', label="Observed")
plt.plot(idx, pred_mean, 'x', label="Predicted mean")
plt.fill_between(idx, lower, upper, alpha=0.25, label="95% predictive interval")

plt.xlabel("Test sample index")
plt.ylabel("Energy (kWh)")
plt.title("Posterior Predictive Intervals on Test Set")
plt.xticks(idx)
plt.legend()
plt.tight_layout()
plt.show()

"""""""""""""""""""""""""""""""""""""""OPTION A: FULL-DATA FIT + PSIS-LOO"""""""""""""""""""""""""""""""""

# Refit on ALL m equations (no held-out split). PSIS-LOO reuses this single
# posterior to estimate out-of-sample error for EVERY equation via importance
# reweighting, so the metrics below are computed over all m points instead of
# one arbitrary 20% test fold — the Bayesian analog of the repeated K-fold CV in
# Tasks_energy_loading_swinging.py, but from a single fit.
# Scale the sigma prior off the full b (not b_train), matching the data fit here.
sigma_prior_config_full = make_sigma_prior_config(b)
model_full, trace_full = fit_bayesian_activity_power(grading,
    A,
    b,
    x_prior_config=x_prior_config,
    sigma_prior_config=sigma_prior_config_full,
    draws=2000,
    tune=2000,
    random_seed=42,
    target_accept=0.9
)

# Pointwise log-likelihood log p(b_i | theta^s) is required for LOO.
pm.compute_log_likelihood(trace_full, model=model_full)

# ---- Headline coefficients from the full-data fit (printed like the test fit) ----
print("\n---------------- FULL-DATA POSTERIOR MEANS ----------------")
print_posterior_means(trace_full, col_present)

# ---- Posterior distributions from the full-data fit (same style/plot as the
# test fit; reuses `present`, `nrows`, `ncols` built in the PLOTTING section) ----
posterior_renamed_full = xr.Dataset({
    lab: trace_full.posterior["x"].sel(activity=coord).reset_coords(drop=True)
    for lab, coord in present
})
axes = az.plot_posterior(
    posterior_renamed_full,
    var_names=[lab for lab, _ in present],
    hdi_prob=0.95,
    grid=(nrows, ncols),
    figsize=(4 * ncols, 3.5 * nrows)
)
label_posterior_axes(axes, posterior_renamed_full, [lab for lab, _ in present])
plt.suptitle("Full-data posterior")
plt.tight_layout()
plt.show()

# ---- az.loo diagnostics (elpd_loo +/- SE, p_loo, Pareto-k) ----
loo_res = az.loo(trace_full, pointwise=True)
print("\n---------------- PSIS-LOO DIAGNOSTICS (full-data fit, all {} equations) ----------------".format(len(b)))
print(loo_res)

khat = np.asarray(loo_res.pareto_k.values)
bad_idx = np.where(khat > 0.7)[0]
if bad_idx.size > 0:
    print(f"\nWARNING: {bad_idx.size} equation(s) have Pareto k > 0.7 (indices {bad_idx.tolist()}).")
    print("Their LOO estimate is unreliable (too influential for importance reweighting).")
    print("Fix: refit without each flagged equation (exact LOO) via az.reloo with a")
    print("PyMC sampling wrapper, or inspect these equations as influential points.")

# ---- LOO point predictions: PSIS-weighted posterior-mean prediction for each
# equation, i.e. what the model predicts for equation i as if it were held out ----
log_lik = trace_full.log_likelihood["b_obs"].stack(sample=("chain", "draw"))
obs_dim = [d for d in log_lik.dims if d != "sample"][0]
log_lik = log_lik.transpose(obs_dim, "sample")                 # (N, S)
log_weights, _ = az.psislw(-log_lik.values)                    # smoothed, normalized per obs
w = np.exp(np.asarray(log_weights))
w /= w.sum(axis=1, keepdims=True)

x_samp = trace_full.posterior["x"].stack(sample=("chain", "draw")) \
             .transpose("sample", "activity").values           # (S, n)
mu_loo = A @ x_samp.T                                           # (N, S): predicted energy per draw
pred_loo = np.sum(w * mu_loo, axis=1)                          # (N,): LOO predictive mean

resid = b - pred_loo
loo_rmse = np.sqrt(np.mean(resid ** 2))
loo_mae  = np.mean(np.abs(resid))
loo_mape = np.mean(np.abs(resid / b)) * 100      # mean |b - b_loo| / |b|
loo_nmae = loo_mae / np.mean(np.abs(b)) * 100    # normalized MAE (MAE / mean|b|)

# Reference points for interpreting the error (see LOO discussion):
#   - posterior sigma: the model's own estimate of irreducible per-equation noise
#   - SoC-quantization floor: irreducible noise on b from 1% SoC resolution across
#     the two bucket endpoints, sqrt(2) * (1% / sqrt(12)) * Battery_cap / 100
sigma_post = trace_full.posterior["sigma"].mean().item()
sigma_quant = np.sqrt(2) * (1.0 / np.sqrt(12)) * Battery_cap / 100

print("\n---------------- LOO TEST METRICS (over all {} equations) ----------------".format(len(b)))
print(f"LOO RMSE:           {loo_rmse:.4f} kWh")
print(f"LOO MAE:            {loo_mae:.4f} kWh")
print(f"LOO MAPE:           {loo_mape:.2f} %   (mean |b - b_loo| / |b|)")
print(f"LOO MAE / mean|b|:  {loo_nmae:.2f} %   (normalized MAE)")
print(f"\nContext:")
print(f"  Posterior sigma (model noise):   {sigma_post:.4f} kWh")
print(f"  SoC-quantization floor on b:     {sigma_quant:.4f} kWh")
print(f"  Mean |b|:                        {np.mean(np.abs(b)):.4f} kWh")

# Observed vs LOO-predicted across ALL equations.
plt.figure(figsize=(6, 6))
plt.scatter(b, pred_loo)
lo = min(np.min(b), np.min(pred_loo))
hi = max(np.max(b), np.max(pred_loo))
plt.plot([lo, hi], [lo, hi], '--')
plt.xlabel("Observed energy (kWh)")
plt.ylabel("LOO-predicted mean energy (kWh)")
plt.title(f"Observed vs LOO-Predicted (all {len(b)} equations)")
plt.tight_layout()
plt.show()

if sys.flags.interactive == 0 and plt.get_fignums():
    plt.ioff()
    plt.show()