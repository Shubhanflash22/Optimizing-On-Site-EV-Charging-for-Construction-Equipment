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


"""""""""""""""""""""""""""""""""""""""October 2025: Site 1: Soil"""""""""""""""""""""""""""""""""

################## October 21 2025: Site 1: Soil

Data_tasks_1 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Oct_21_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

################## October 22 2025: Site 1: Soil

Data_tasks_2 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Oct_22_Tasks_1.xlsx',
    sheet_name="Sheet1"
)
Data_tasks_3 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Oct_22_Tasks_2.xlsx',
    sheet_name="Sheet1"
)
Data_tasks_4 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Oct_22_Tasks_3.xlsx',
    sheet_name="Sheet1"
)
Data_tasks_5 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Oct_22_Tasks_4.xlsx',
    sheet_name="Sheet1"
)
Data_tasks_6 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Oct_22_Tasks_5.xlsx',
    sheet_name="Sheet1"
)

################## October 23 2025: Site 1: Soil

Data_tasks_7 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Oct_23_Tasks_1.xlsx',
    sheet_name="Sheet1"
)


"""""""""""""""""""""""""""""""""""""""February 2026: Site 1: Soil"""""""""""""""""""""""""""""""""

################## February 02 2026: Site 1: Soil

Data_tasks_8 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_02_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_9 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_02_Tasks_2.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_10 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_02_Tasks_3.xlsx',
    sheet_name="Sheet1"
)



################## February 03 2026: Site 1: Soil


Data_tasks_11 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_03_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_12 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_03_Tasks_2.xlsx',
    sheet_name="Sheet1"
)


################## February 04 2026: Site 1: Concrete


Data_tasks_13 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_04_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_14 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_04_Tasks_2.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_15 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_04_Tasks_3.xlsx',
    sheet_name="Sheet1"
)

################## February 11 2026: Site 1: Concrete

Data_tasks_16 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_11_Tasks_2.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_17 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_11_Tasks_3.xlsx',
    sheet_name="Sheet1"
)

################## February 11 2026: Site 2: Sand


Data_tasks_18 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_11_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

################## February 12 2026: Site 2: Sand


Data_tasks_19 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_12_Tasks_1.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_20 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_12_Tasks_2.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_21 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_12_Tasks_3.xlsx',
    sheet_name="Sheet1"
)

Data_tasks_22 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_12_Tasks_4.xlsx',
    sheet_name="Sheet1"
)

################## February 13 2026: Site 2: Sand

Data_tasks_23 = pd.read_excel(
    r'/Users/avikghosh/Desktop/UCSD Postdoc/Code/Optimizing-On-Site-EV-Charging-for-Construction-Equipment/7.CVAT (excel to xml)/Analysis/Feb_13_Tasks_1.xlsx',
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
    Build A and b using 4 columns:
    [Digging, Loading+Swinging, Travelling, Idling]
    """
    A_rows_all = []
    b_rows_all = []

    if (grading == "False"):
        
        i = 0

        for j in range(1, len(df_clean)):
            soc_now  = df_clean.iloc[j]["SoC"]
            soc_prev = df_clean.iloc[j - 1]["SoC"]

            if pd.notna(soc_now) and pd.notna(soc_prev) and (soc_now - soc_prev != 0):
                df_slice = df_clean.iloc[i:j+1]

                total_s_Digging    = df_slice.loc[df_slice["Activity"] == "Digging", "duration_s"].sum()
                total_s_Grading_1    = df_slice.loc[df_slice["Activity"] == "Grading 1", "duration_s"].sum()
                total_s_Loading    = df_slice.loc[df_slice["Activity"] == "Loading", "duration_s"].sum()
                total_s_Swinging   = df_slice.loc[df_slice["Activity"] == "Swinging", "duration_s"].sum()
                total_s_Grading_2   = df_slice.loc[df_slice["Activity"] == "Grading 2", "duration_s"].sum()
                total_s_Travelling = df_slice.loc[df_slice["Activity"] == "Travelling", "duration_s"].sum()
                total_s_Idling     = df_slice.loc[df_slice["Activity"] == "Idling", "duration_s"].sum()
                total_s_Mixing = df_slice.loc[df_slice["Activity"] == "Mixing", "duration_s"].sum()


                total_energy = -(soc_now - soc_prev) * battery_cap / 100

                A_row = [
                    total_s_Digging/3600 + total_s_Grading_1/3600,
                    total_s_Loading/3600 + total_s_Swinging/3600 + total_s_Grading_2/3600,
                    total_s_Travelling/3600,
                    total_s_Idling/3600,
                    total_s_Mixing/3600
                ]
                b_row = [total_energy]

                A_rows_all.append(A_row)
                b_rows_all.append(b_row)

                i = j + 1
    
    else:
        
         i = 0

         for j in range(1, len(df_clean)):
             soc_now  = df_clean.iloc[j]["SoC"]
             soc_prev = df_clean.iloc[j - 1]["SoC"]

             if pd.notna(soc_now) and pd.notna(soc_prev) and (soc_now - soc_prev != 0):
                 df_slice = df_clean.iloc[i:j+1]

                 total_s_Digging    = df_slice.loc[df_slice["Activity"] == "Digging", "duration_s"].sum()
                 total_s_Grading_1    = df_slice.loc[df_slice["Activity"] == "Grading 1", "duration_s"].sum()
                 total_s_Loading    = df_slice.loc[df_slice["Activity"] == "Loading", "duration_s"].sum()
                 total_s_Swinging   = df_slice.loc[df_slice["Activity"] == "Swinging", "duration_s"].sum()
                 total_s_Grading_2   = df_slice.loc[df_slice["Activity"] == "Grading 2", "duration_s"].sum()
                 total_s_Travelling = df_slice.loc[df_slice["Activity"] == "Travelling", "duration_s"].sum()
                 total_s_Idling     = df_slice.loc[df_slice["Activity"] == "Idling", "duration_s"].sum()
                 total_s_Mixing = df_slice.loc[df_slice["Activity"] == "Mixing", "duration_s"].sum()


                 total_energy = -(soc_now - soc_prev) * battery_cap / 100

                 A_row = [
                     total_s_Digging/3600,
                     total_s_Grading_1/3600,
                     total_s_Loading/3600 + total_s_Swinging/3600,
                     total_s_Grading_2/3600,
                     total_s_Travelling/3600,
                     total_s_Idling/3600,
                     total_s_Mixing/3600
                 ]
                 b_row = [total_energy]

                 A_rows_all.append(A_row)
                 b_rows_all.append(b_row)

                 i = j + 1

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


def print_posterior_means(trace):
    x_post_mean = trace.posterior["x"].mean(dim=("chain", "draw")).values
    sigma_post_mean = trace.posterior["sigma"].mean(dim=("chain", "draw")).values.item()
    
    if (grading == "False"):
        print(f"Digging:            {x_post_mean[0]:.3f} kW")
        print(f"Loading+Swinging:   {x_post_mean[1]:.3f} kW")
        print(f"Traveling:          {x_post_mean[2]:.3f} kW")
        print(f"Idling:             {x_post_mean[3]:.3f} kW")
        print(f"Mixing:             {x_post_mean[4]:.3f} kW")
        print(f"Sigma:              {sigma_post_mean:.3f}")
    else:
        print(f"Digging:   {x_post_mean[0]:.3f} kW")
        print(f"Grading 1:   {x_post_mean[1]:.3f} kW")
        print(f"Loading+Swinging:   {x_post_mean[2]:.3f} kW")
        print(f"Grading 2:   {x_post_mean[3]:.3f} kW")
        print(f"Traveling: {x_post_mean[4]:.3f} kW")
        print(f"Idling:    {x_post_mean[5]:.3f} kW")
        print(f"Mixing:             {x_post_mean[6]:.3f} kW")
        print(f"Sigma:     {sigma_post_mean:.3f}")


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
Data_tasks_clean_15 = prepare_task_data(Data_tasks_15)      # Feb 04, 2026: Concrete

Data_tasks_clean_16 = prepare_task_data(Data_tasks_16)
Data_tasks_clean_17 = prepare_task_data(Data_tasks_17)      # Feb 11, 2026: Concrete

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


"""""""""""""""""""""""""""""""""""""""FORM EQUATIONS OF TASKS OF ALL DAYS"""""""""""""""""""""""""""""""""

all_task_dfs = [
    Data_tasks_clean_1,     # Oct 21, 2025: Soil
    
    Data_tasks_clean_2,
    Data_tasks_clean_3,
    Data_tasks_clean_4,
    Data_tasks_clean_5,
    Data_tasks_clean_6,
    Data_tasks_clean_7,     # Oct 22-23, 2025: Soil
    
    Data_tasks_clean_8,
    Data_tasks_clean_9,
    Data_tasks_clean_10,    # Feb 02, 2026: Soil
    
    Data_tasks_clean_11,
    Data_tasks_clean_12,    # Feb 03, 2026: Soil
    
    # Data_tasks_clean_13,
    # Data_tasks_clean_14,
    # Data_tasks_clean_15, 
    # Data_tasks_clean_16,
    # Data_tasks_clean_17,    # Feb 04 and Feb 11, 2026: Concrete
    
    # Data_tasks_clean_18,
    # Data_tasks_clean_19,
    # Data_tasks_clean_20, 
    # Data_tasks_clean_21,
    # Data_tasks_clean_22, 
    # Data_tasks_clean_23,    # Feb 11-13, 2026: Sand
    ]

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

# Time_Digging = np.sum(A[:, [0]], axis=0)
# Time_Loading_Swinging = np.sum(A[:, [1]], axis=0)
# Time_Traveling = np.sum(A[:, [2]], axis=0)

# Time_all = Time_Digging + Time_Loading_Swinging + Time_Traveling

"""""""""""""""""""""""""""""""""""""""SPLIT TRAINING AND TESTING"""""""""""""""""""""""""""""""""

A_train, A_test, b_train, b_test = split_train_test(A, b, test_size=0.15)

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
# ---------- Option 2: TruncatedNormal prior on x ----------
if (grading == "False"):

    x_prior_config = {
         "dist": "truncated_normal",
         "mu": np.array([4.6, 3, 4.5, 0, 4.7]),
         "sigma": np.array([1.0, 1.0, 1.5, 0.1, 0.1]),
        "lower": 0.0
    }    
else:
    x_prior_config = {
         "dist": "truncated_normal",
         "mu": np.array([0, 5.5, 3.5, 5.4, 1.0, 0, 4.7]),
         "sigma": np.array([0.1, 1, 1.0, 1, 1.5, 0.1, 0.1]),
        "lower": 0.0
    }       

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

sigma_prior_config = {
    "dist": "halfnormal",
    "sigma": float(np.std(b_train))
}

# Alternatives:
# sigma_prior_config = {"dist": "exponential", "lam": 1.0 / max(float(np.std(b_train)), 1e-6)}
# sigma_prior_config = {"dist": "halfstudentt", "sigma": float(np.std(b_train)), "nu": 4}

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

print_posterior_means(trace)

# az.plot_posterior(trace, var_names=["x", "sigma"], hdi_prob=0.95)

if (grading == "False"):

    posterior_renamed = xr.Dataset({
        "Digging": trace.posterior["x"].sel(activity="Digging").reset_coords(drop=True),
        "Loading+Swinging": trace.posterior["x"].sel(activity="Loading + Swinging").reset_coords(drop=True),
        "Traveling": trace.posterior["x"].sel(activity="Traveling").reset_coords(drop=True),
        "Idling": trace.posterior["x"].sel(activity="Idling").reset_coords(drop=True),
        "Mixing": trace.posterior["x"].sel(activity="Mixing").reset_coords(drop=True),

    })

    axes = az.plot_posterior(
        posterior_renamed,
        var_names=["Digging", "Loading+Swinging", "Traveling", "Idling", "Mixing"],
        hdi_prob=0.95,
        grid=(2, 3),
        figsize=(12, 7)
    )

    for ax in np.ravel(axes):
        if ax is not None:
            ax.set_xlabel("Power (kW)")

    plt.tight_layout()
    plt.show()
    
else:
    
    posterior_renamed = xr.Dataset({
        "Digging": trace.posterior["x"].sel(activity="Digging").reset_coords(drop=True),
        "Grading 1": trace.posterior["x"].sel(activity="Grading 1").reset_coords(drop=True),
        "Loading+Swinging": trace.posterior["x"].sel(activity="Loading + Swinging").reset_coords(drop=True),
        "Grading 2": trace.posterior["x"].sel(activity="Grading 2").reset_coords(drop=True),
        "Traveling": trace.posterior["x"].sel(activity="Traveling").reset_coords(drop=True),
        "Idling": trace.posterior["x"].sel(activity="Idling").reset_coords(drop=True),
        "Mixing": trace.posterior["x"].sel(activity="Mixing").reset_coords(drop=True)

    })

    axes = az.plot_posterior(
        posterior_renamed,
        var_names=["Digging", "Grading 1", "Loading+Swinging", "Grading 2", "Traveling", "Idling", "Mixing"],
        hdi_prob=0.95,
        grid=(3, 3),
        figsize=(12, 7)
    )

    for ax in np.ravel(axes):
        if ax is not None:
            ax.set_xlabel("Power (kW)")

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

if sys.flags.interactive == 0 and plt.get_fignums():
    plt.ioff()
    plt.show()