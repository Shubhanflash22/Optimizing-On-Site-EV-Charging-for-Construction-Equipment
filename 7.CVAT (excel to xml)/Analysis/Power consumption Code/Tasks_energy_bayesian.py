#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  6 20:50:47 2026

@author: avikghosh
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 01:23:12 2025

@author: avikghosh
"""

"""""""""""""""""""""""""""""""""""""""IMPORT PACKAGES HERE"""""""""""""""""""""""""""""""""

from IPython import get_ipython
ip = get_ipython()
if ip is not None:
    ip.run_line_magic("reset", "-f")  # clears all variables

import os
os.system('clear')

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
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
import xarray as xr
import arviz as az

start_time = time.time()

"""""""""""""""""""""""""""""""""""""""READ INPUT DATA HERE"""""""""""""""""""""""""""""""""

grading = "True";

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
    For each interval where SoC changes, sum the duration spent in each activity
    between two SoC change points, and build one row of A and b.

    Returns
    -------
    A_rows_all : list of lists
        Each row is [Digging, Loading, Swinging, Travelling, Idling] in hours.
    b_rows_all : list of lists
        Each row is [total_energy] in kWh.
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

                total_energy = -(soc_now - soc_prev) * battery_cap / 100

                A_row = [
                    total_s_Digging/3600 + total_s_Grading_1/3600,
                    total_s_Loading/3600,
                    total_s_Swinging/3600 + total_s_Grading_2/3600,
                    total_s_Travelling/3600,
                    total_s_Idling/3600
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

                 total_energy = -(soc_now - soc_prev) * battery_cap / 100

                 A_row = [
                     total_s_Digging/3600,
                     total_s_Grading_1/3600,
                     total_s_Loading/3600,
                     total_s_Swinging/3600,
                     total_s_Grading_2/3600,
                     total_s_Travelling/3600,
                     total_s_Idling/3600
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
    Convert desired mean and std to Gamma(alpha, beta) parameters
    using PyMC's alpha-beta parameterization:
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
    Build prior for x based on prior_config.

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
    Build prior for sigma based on prior_config.

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
    target_accept=0.9,
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
        
        activity_names = ["Digging", "Loading", "Swinging", "Traveling", "Idling"]

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
                random_seed=random_seed,
                target_accept=target_accept,
                return_inferencedata=True
            )
            
    else:
        
        activity_names = ["Digging", "Grading 1", "Loading", "Swinging", "Grading 2", "Traveling", "Idling"]

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
                random_seed=random_seed,
                target_accept=target_accept,
                return_inferencedata=True
            )

    return model, trace


def predictive_interval_coverage_from_trace(A_test, b_test, trace, alpha=0.05, random_seed=42):
    """
    Compute posterior predictive mean, predictive intervals, and coverage
    using posterior samples stored in a PyMC/ArviZ trace.
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

        print(f"Digging:   {x_post_mean[0]:.3f} kW")
        print(f"Loading:   {x_post_mean[1]:.3f} kW")
        print(f"Swinging:  {x_post_mean[2]:.3f} kW")
        print(f"Traveling: {x_post_mean[3]:.3f} kW")
        print(f"Idling:    {x_post_mean[4]:.3f} kW")
        print(f"Sigma:     {sigma_post_mean:.3f}")
    
    else:
        
        print(f"Digging:   {x_post_mean[0]:.3f} kW")
        print(f"Grading 1:   {x_post_mean[1]:.3f} kW")
        print(f"Loading:   {x_post_mean[2]:.3f} kW")
        print(f"Swinging:  {x_post_mean[3]:.3f} kW")
        print(f"Grading 2:   {x_post_mean[4]:.3f} kW")
        print(f"Traveling: {x_post_mean[5]:.3f} kW")
        print(f"Idling:    {x_post_mean[6]:.3f} kW")
        print(f"Sigma:     {sigma_post_mean:.3f}")

"""""""""""""""""""""""""""""""""""""""PREPARE DATA"""""""""""""""""""""""""""""""""

Data_tasks_clean_1 = prepare_task_data(Data_tasks_1)
Data_tasks_clean_2 = prepare_task_data(Data_tasks_2)
Data_tasks_clean_3 = prepare_task_data(Data_tasks_3)
Data_tasks_clean_4 = prepare_task_data(Data_tasks_4)
Data_tasks_clean_5 = prepare_task_data(Data_tasks_5)
Data_tasks_clean_6 = prepare_task_data(Data_tasks_6)
Data_tasks_clean_7 = prepare_task_data(Data_tasks_7)

Data_tasks_clean_8 = prepare_task_data(Data_tasks_8)
Data_tasks_clean_9 = prepare_task_data(Data_tasks_9)
Data_tasks_clean_10 = prepare_task_data(Data_tasks_10)

Data_tasks_clean_11 = prepare_task_data(Data_tasks_11)
Data_tasks_clean_12 = prepare_task_data(Data_tasks_12)


Data_tasks_clean = [
    Data_tasks_clean_1, Data_tasks_clean_2, Data_tasks_clean_3, Data_tasks_clean_4,
    Data_tasks_clean_5, Data_tasks_clean_6, Data_tasks_clean_7, Data_tasks_clean_8,
    Data_tasks_clean_9, Data_tasks_clean_10, Data_tasks_clean_11, Data_tasks_clean_12,
]

Data_tasks_clean_combined = pd.concat(Data_tasks_clean, ignore_index=True)

"""""""""""""""""""""""""""""""""""""""FORM EQUATIONS OF TASKS OF ALL DAYS"""""""""""""""""""""""""""""""""

all_task_dfs = [
    Data_tasks_clean_1,
    Data_tasks_clean_2,
    Data_tasks_clean_3,
    Data_tasks_clean_4,
    Data_tasks_clean_5,
    Data_tasks_clean_6,
    Data_tasks_clean_7,

    Data_tasks_clean_8,
    Data_tasks_clean_9,
    Data_tasks_clean_10,

    Data_tasks_clean_11,
    Data_tasks_clean_12,

]

A = []
b = []

for df_clean in all_task_dfs:
    A_part, b_part = build_equations_from_tasks(df_clean, Battery_cap, grading)
    A.extend(A_part)
    b.extend(b_part)

df = pd.concat(all_task_dfs)
unique_tasks = df["Activity"].unique()
print(f"The number of unique tasks are in the dataset are: {unique_tasks}")

"""""""""""""""""""""""""""""""""""""""DEFINING MATRIX AND VECTOR"""""""""""""""""""""""""""""""""

A = np.asarray(np.array(A), dtype=float)
b = np.asarray(np.array(b), dtype=float).reshape(-1)

m, n = A.shape

# Time_Digging = np.sum(A[:, [0]], axis=0)
# Time_Loading = np.sum(A[:, [1]], axis=0)
# Time_Swinging = np.sum(A[:, [2]], axis=0)
# Time_Traveling = np.sum(A[:, [3]], axis=0)

# Time_all = Time_Digging + Time_Loading + Time_Swinging + Time_Traveling

"""""""""""""""""""""""""""""""""""""""SPLIT TRAINING AND TESTING"""""""""""""""""""""""""""""""""

A_train, A_test, b_train, b_test = split_train_test(A, b, test_size=0.2)

"""""""""""""""""""""""""""""""""""""""CHOOSE PRIOR SETUP HERE"""""""""""""""""""""""""""""""""

# ---------- Option 1: HalfNormal prior on x ----------

# if (grading == "False"):
#     x_prior_config = {
#         "dist": "halfnormal",
#         "sigma": np.array([5.6, 15, 1.1, 6.5, 0.00001])
#     }
# else:
#     x_prior_config = {
#         "dist": "halfnormal",
#         "sigma": np.array([5.7, 4.4, 15, 1.0, 3.7, 6.5, 0.00001])
#     }

# ---------- Option 2: TruncatedNormal prior on x ----------

if (grading == "False"):
    x_prior_config = {
        "dist": "truncated_normal",
        "mu": np.array([4.5, 11.6, 1.0, 5.1, 0]),
        "sigma": np.array([1.0, 5.0, 2.0, 1.5, 0.1]),
        "lower": 0.0
    }
else:
    x_prior_config = {
        "dist": "truncated_normal",
        "mu": np.array([4.6, 3.5, 13.6, 0.29, 2.93, 5.2, 0]),
        "sigma": np.array([1.0, 1, 5.0, 3.0,1, 1.5, 0.1]),
        "lower": 0.0
    }

# ---------- Option 3: LogNormal prior on x ----------
# x_prior_config = {
#     "dist": "lognormal",
#     "mu": np.log(np.array([5.4, 17.5, 1.0, 8.8, 1.0])),
#     "sigma": np.array([0.35, 0.35, 0.50, 0.35, 0.50])
# }

# ---------- Option 4: Gamma prior on x ----------
# prior_mean = np.array([5.4, 17.5, 1.0, 8.8, 1.0])
# prior_std  = np.array([2.0,  4.0, 0.8, 3.0, 0.5])
# alpha_x, beta_x = gamma_from_mean_std(prior_mean, prior_std)
# x_prior_config = {
#     "dist": "gamma",
#     "alpha": alpha_x,
#     "beta": beta_x
# }

# ---------- Sigma prior ----------
sigma_prior_config = {
    "dist": "halfnormal",
    "sigma": float(np.std(b_train))
}

# Alternative sigma priors:
# sigma_prior_config = {"dist": "exponential", "lam": 1.0 / max(float(np.std(b_train)), 1e-6)}
# sigma_prior_config = {"dist": "halfstudentt", "sigma": float(np.std(b_train)), "nu": 4}

"""""""""""""""""""""""""""""""""""""""BAYESIAN REGRESSION"""""""""""""""""""""""""""""""""

model, trace = fit_bayesian_activity_power( grading,
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
        "Loading": trace.posterior["x"].sel(activity="Loading").reset_coords(drop=True),
        "Swinging": trace.posterior["x"].sel(activity="Swinging").reset_coords(drop=True),
        "Traveling": trace.posterior["x"].sel(activity="Traveling").reset_coords(drop=True),
        "Idling": trace.posterior["x"].sel(activity="Idling").reset_coords(drop=True)
    })

    axes = az.plot_posterior(
        posterior_renamed,
        var_names=["Digging", "Loading", "Swinging", "Traveling", "Idling"],
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
        "Loading": trace.posterior["x"].sel(activity="Loading").reset_coords(drop=True),
        "Swinging": trace.posterior["x"].sel(activity="Swinging").reset_coords(drop=True),
        "Grading 2": trace.posterior["x"].sel(activity="Grading 2").reset_coords(drop=True),
        "Traveling": trace.posterior["x"].sel(activity="Traveling").reset_coords(drop=True),
        "Idling": trace.posterior["x"].sel(activity="Idling").reset_coords(drop=True)
    })

    axes = az.plot_posterior(
        posterior_renamed,
        var_names=["Digging", "Grading 1", "Loading", "Swinging", "Grading 2", "Traveling", "Idling"],
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

print("\n---------------- TEST RESULTS ----------------")
print(f"Test RMSE: {rmse:.4f} kWh")
print(f"Test MAE: {mae:.4f} kWh")
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