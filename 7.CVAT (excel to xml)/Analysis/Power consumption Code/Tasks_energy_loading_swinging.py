#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 23 01:23:12 2025

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
    For each interval where SoC changes, sum the duration spent in each activity
    between two SoC change points, and build one row of A and b.

    Returns
    -------
    A_rows_all : list of lists
        Each row is [Digging, Loading, Swinging, Travelling, Idling] in seconds.
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
                     total_s_Mixing/3600,
                 ]
                 b_row = [total_energy]

                 A_rows_all.append(A_row)
                 b_rows_all.append(b_row)

                 i = j + 1

    return A_rows_all, b_rows_all


def split_train_test(A, b, test_size=0.2, random_state=42, shuffle=True):
    A_train, A_test, b_train, b_test = train_test_split(
        A, b,
        test_size=test_size,
        random_state=random_state,
        shuffle=shuffle
    )
    return A_train, A_test, b_train, b_test


def solve_activity_power(A, b, n, W, reg_param, grading):
    
    if (grading == "False"):
        
        z = cp.Variable(n, nonneg=True)
        constraints = [z[3] == 0]   # Idling power fixed to 0

        objective = cp.Minimize(cp.sum_squares(cp.sqrt(W) @ (A @ z - b)) + reg_param*cp.sum_squares(z))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)

    else:
         
        z = cp.Variable(n, nonneg=True)
        constraints = [z[5] == 0]   # Idling power fixed to 0

        objective = cp.Minimize(cp.sum_squares(cp.sqrt(W) @ (A @ z - b)) + reg_param*cp.sum_squares(z))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)
    
    return z.value, objective.value
    

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

A = np.array(A)
b = np.array(b)

A = np.asarray(A)
b = np.asarray(b).reshape(-1)

m, n = A.shape

if (grading == "False"):

    Time_Digging = np.sum(A[:, [0]], axis = 0); #axis = 0 means across rows
    Time_Loading_Swinging = np.sum(A[:, [1]], axis = 0);
    Time_Traveling = np.sum(A[:, [2]], axis = 0);
    Time_Mixing = np.sum(A[:, [4]], axis = 0);


    Time_all = Time_Digging + Time_Loading_Swinging + Time_Traveling + Time_Mixing;

    A_drop= np.delete(A, 3, axis=1) #axis = 1 means across columns

    df = pd.DataFrame(A)
    df.columns = ['Digging', 'Loading+Swinging', 'Traveling', 'Idling', 'Mixing']
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.xticks(rotation=45)
    plt.show()
    
else:
    
    Time_Digging = np.sum(A[:, [0]], axis = 0); #axis = 0 means across rows
    Time_Grading_1 = np.sum(A[:, [1]], axis = 0);
    Time_Loading_Swinging = np.sum(A[:, [2]], axis = 0);
    Time_Grading_2 = np.sum(A[:, [3]], axis = 0);
    Time_Traveling = np.sum(A[:, [4]], axis = 0);
    Time_Mixing = np.sum(A[:, [6]], axis = 0);


    Time_all = Time_Digging + Time_Grading_1 + Time_Loading_Swinging + Time_Grading_2 + Time_Traveling + Time_Mixing;

    A_drop= np.delete(A, 5, axis=1) #axis = 1 means across columns; Here delete idling column

    df = pd.DataFrame(A)
    df.columns = ['Digging', 'Grading 1', 'Loading+Swinging', 'Grading 2', 'Traveling', 'Idling', 'Mixing']
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.xticks(rotation=45)
    plt.show()

# sum_A_drop = np.sum(A_drop, axis=1); # Sum time of all activites per SOC change in rows
# mean_sum = np.mean(sum_A_drop)
# W = np.diag(sum_A_drop/mean_sum)




"""""""""""""""""""""""""""""""""""""""TRAIN / TEST SPLIT"""""""""""""""""""""""""""""""""

A_train, A_test, b_train, b_test = split_train_test(
    A, b, test_size=0.2, random_state=42, shuffle=True
)

# b_ref_train = statistics.median(b_train)
# weight_soc_train = np.minimum(np.abs(b_train) / b_ref_train, 1).flatten()
# weight_soc_train = weight_soc_train / np.mean(weight_soc_train)
# W_train = np.diag(weight_soc_train)

m_train = A_train.shape[0]
W_train = np.eye(m_train)

reg_param = 0e-3

"""""""""""""""""""""""""""""""""""""""CVXPY (TRAIN)"""""""""""""""""""""""""""""""""

z, objective = solve_activity_power(A_train, b_train, n, W_train, reg_param, grading)



"""""""""""""""""""""""""""""""""""""""PRINTING"""""""""""""""""""""""""""""""""

if (grading == "False"):

    print(f"\nObjective with weighing and regularization {reg_param} is:  {objective:.5f}\n")

    print(f"\nDigging:   {z[0]:.2f} kW")
    print(f"\nLoading+Swinging:   {z[1]:.2f} kW")
    print(f"\nTraveling:  {z[2]:.2f} kW")
    print(f"\nIdling: {z[3]:.2f} kW")
    print(f"\nMixing: {z[4]:.2f} kW")


    print(f"\n\nDigging:   {Time_Digging[0]*100/Time_all[0]:.0f}%")
    print(f"\nLoading + Swinging:   {Time_Loading_Swinging[0]*100/Time_all[0]:.0f}%")
    print(f"\nTraveling: {Time_Traveling[0]*100/Time_all[0]:.0f}%")
    print(f"\nMixing: {Time_Mixing[0]*100/Time_all[0]:.0f}%")
    
else:
    
    print(f"\nObjective with weighing and regularization {reg_param} is:  {objective:.5f}\n")

    print(f"\nDigging:   {z[0]:.2f} kW")
    print(f"\nGrading 1:   {z[1]:.2f} kW")
    print(f"\nLoading + Swinging:   {z[2]:.2f} kW")
    print(f"\nGrading 2:  {z[3]:.2f} kW")
    print(f"\nTraveling: {z[4]:.2f} kW")
    print(f"\nIdling:    {z[5]:.2f} kW")
    print(f"\nMixing: {z[6]:.2f} kW")

    
    print(f"\n\nDigging:   {Time_Digging[0]*100/Time_all[0]:.0f}%")
    print(f"\nGrading 1:   {Time_Grading_1[0]*100/Time_all[0]:.0f}%")
    print(f"\nLoading + Swinging:   {Time_Loading_Swinging[0]*100/Time_all[0]:.0f}%")
    print(f"\nGrading 2:   {Time_Grading_2[0]*100/Time_all[0]:.0f}%")
    print(f"\nTraveling: {Time_Traveling[0]*100/Time_all[0]:.0f}%")
    print(f"\nMixing: {Time_Mixing[0]*100/Time_all[0]:.0f}%\n")


"""""""""""""""""""""""""""""""""""""""TEST EVALUATION"""""""""""""""""""""""""""""""""

b_pred_test = A_test @ z

mae_test = mean_absolute_error(b_test, b_pred_test)
mse_test = mean_squared_error(b_test, b_pred_test)
rmse_test = np.sqrt(mse_test)

# MAE as a percentage of the observed b_test.
# MAPE: average of |b - b_pred| / |b| (per-sample relative error)
mape_test = np.mean(np.abs((b_test - b_pred_test) / b_test)) * 100
# Normalized MAE: MAE divided by mean(|b_test|). More stable when b_test has
# values close to zero (avoids exploding per-sample ratios).
nmae_test = mae_test / np.mean(np.abs(b_test)) * 100

print("\n---------------- TEST RESULTS ----------------")
print(f"Test MAE:           {mae_test:.4f} kWh")
print(f"Test RMSE:          {rmse_test:.4f} kWh")
print(f"Test MAPE:          {mape_test:.2f} %   (mean |b - b_pred| / |b|)")
print(f"Test MAE / mean|b|: {nmae_test:.2f} %   (MAE relative to average |b|)")

results_df = pd.DataFrame({
    "Observed_b_test":  b_test,
    "Predicted_b_test": b_pred_test,
})

print("\nTest prediction summary:")
print(results_df)

# Observed vs Predicted on the test set
plt.figure(figsize=(6, 6))
plt.scatter(b_test, b_pred_test)
min_val = min(np.min(b_test), np.min(b_pred_test))
max_val = max(np.max(b_test), np.max(b_pred_test))
plt.plot([min_val, max_val], [min_val, max_val], '--')
plt.xlabel("Observed test energy (kWh)")
plt.ylabel("Predicted test energy (kWh)")
plt.title("Observed vs Predicted on Test Set")
plt.tight_layout()
plt.show()

# Per-sample line plot on the test set
plt.figure(figsize=(10, 5))
idx = np.arange(1, len(b_test) + 1)
plt.plot(idx, b_test, 'o', label="Observed")
plt.plot(idx, b_pred_test, 'x', label="Predicted")
plt.xlabel("Test sample index")
plt.ylabel("Energy (kWh)")
plt.title("Test Set: Observed vs Predicted")
plt.xticks(idx)
plt.legend()
plt.tight_layout()
plt.show()


# """""""""""""""""""""""""""""""""""""""POST ANALYSIS 1"""""""""""""""""""""""""""""""""


# power_estimated = np.array([Digging_power, Loading_power, Swinging_power, Traveling_power, Idling_power])

# energy_estimated = A@power_estimated.T/3600;

# mae = mean_absolute_error(b, energy_estimated);
# mse = mean_squared_error(b, energy_estimated);
# rmse = np.sqrt(mse);

# print(f"\nMean Absolute Error (MAE): {mae:.5f}")
# print(f"\nRoot Mean Squared Error (RMSE): {rmse:.5f}")


# """""""""""""""""""""""""""""""""""""""POST ANALYSIS 2"""""""""""""""""""""""""""""""""

# #power_estimated_2 = np.array([4.25972450, 1.36857348e+01, 0, 7.24081004, 0])
# #power_estimated_2 = np.array([4.32613791, 1.41901819e+01, 0, 7.09332382, 0])
# power_estimated_2 = np.array([4.34243878, 4.09453713, 3.08962349, 6.4902995, 0])
# #power_estimated_2 = np.array([4.384353783376105, 2.3037187931958103, 3.666943718435899, 6.133078484893416, 0])

# energy_estimated_2 = A@power_estimated_2.T/3600;

# mae = mean_absolute_error(b, energy_estimated_2);
# mse = mean_squared_error(b, energy_estimated_2);
# rmse = np.sqrt(mse);

# print(f"\nMean Absolute Error (MAE) 2: {mae:.5f}")
# print(f"\nRoot Mean Squared Error (RMSE) 2: {rmse:.5f}")

"""""""""""""""""""""""""""""""""""""""PLOTTING"""""""""""""""""""""""""""""""""

# labels = ['Digging', 'Loading+Swinging', 'Traveling', 'Idling']
# corr = np.corrcoef(A, rowvar=False)

# plt.figure(figsize=(6, 5))
# plt.imshow(corr, interpolation='nearest')
# plt.colorbar(label='Correlation')
# plt.xticks(range(len(labels)), labels, rotation=45)
# plt.yticks(range(len(labels)), labels)
# plt.title('Activity Correlation Matrix')

# for i in range(corr.shape[0]):
#     for j in range(corr.shape[1]):
#         plt.text(j, i, f"{corr[i, j]:.2f}", ha='center', va='center')

# plt.tight_layout()
# plt.show()

if sys.flags.interactive == 0 and plt.get_fignums():
    plt.ioff()
    plt.show()