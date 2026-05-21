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


def solve_activity_power(A, b, n, W, reg_param, grading):
    
    
    if (grading == "False"):
        
        z = cp.Variable(n, nonneg=True)
        constraints = [z[4] == 0]   # Idling power fixed to 0

        objective = cp.Minimize(cp.sum_squares(cp.sqrt(W) @ (A @ z - b)) + reg_param*cp.sum_squares(z))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)

    else:
         
        z = cp.Variable(n, nonneg=True)
        constraints = [z[6] == 0]   # Idling power fixed to 0

        objective = cp.Minimize(cp.sum_squares(cp.sqrt(W) @ (A @ z - b)) + reg_param*cp.sum_squares(z))

        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.MOSEK)

    
    return z.value, objective.value
    
"""""""""""""""""""""""""""""""""""""""PREPARE DATA"""""""""""""""""""""""""""""""""

Data_tasks_clean_1 = prepare_task_data(Data_tasks_1)
Data_tasks_clean_2 = prepare_task_data(Data_tasks_2)
Data_tasks_clean_3 = prepare_task_data(Data_tasks_3)
Data_tasks_clean_4 = prepare_task_data(Data_tasks_4)
Data_tasks_clean_5 = prepare_task_data(Data_tasks_5)   # fixed
Data_tasks_clean_6 = prepare_task_data(Data_tasks_6)   # fixed
Data_tasks_clean_7 = prepare_task_data(Data_tasks_7)   # fixed

Data_tasks_clean_8 = prepare_task_data(Data_tasks_8)   # fixed
Data_tasks_clean_9 = prepare_task_data(Data_tasks_9)   # fixed
Data_tasks_clean_10 = prepare_task_data(Data_tasks_10)   # fixed


Data_tasks_clean_11 = prepare_task_data(Data_tasks_11)   # fixed
Data_tasks_clean_12 = prepare_task_data(Data_tasks_12)   # fixed





Data_tasks_clean = [Data_tasks_clean_1,Data_tasks_clean_2, Data_tasks_clean_3, Data_tasks_clean_4, Data_tasks_clean_5, Data_tasks_clean_6, 
                    Data_tasks_clean_7,Data_tasks_clean_8, Data_tasks_clean_9, Data_tasks_clean_10, Data_tasks_clean_11, Data_tasks_clean_12, 
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
print(f"\n\n************OUTPUT************\n\n")
print(f"The number of unique tasks are in the dataset are: {unique_tasks}")

# A_no_swing = np.delete(A, 2, axis=1)
#np.linalg.cond(A)
"""""""""""""""""""""""""""""""""""""""DEFINING MATRIX AND VECTOR"""""""""""""""""""""""""""""""""

A = np.array(A)
b = np.array(b)

A = np.asarray(A)
b = np.asarray(b).reshape(-1)

m, n = A.shape

if (grading == "False"):

    Time_Digging = np.sum(A[:, [0]], axis = 0); #axis = 0 means across rows
    Time_Loading = np.sum(A[:, [1]], axis = 0);
    Time_Swinging = np.sum(A[:, [2]], axis = 0);
    Time_Traveling = np.sum(A[:, [3]], axis = 0);

    Time_all = Time_Digging + Time_Loading + Time_Swinging + Time_Traveling;

    A_drop= np.delete(A, 4, axis=1) #axis = 1 means across columns; Here delete idling column

    df = pd.DataFrame(A)
    df.columns = ['Digging', 'Loading', 'Swinging', 'Traveling', 'Idling']
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.xticks(rotation=45)
    plt.show()
    
else:
    
    Time_Digging = np.sum(A[:, [0]], axis = 0); #axis = 0 means across rows
    Time_Grading_1 = np.sum(A[:, [1]], axis = 0);
    Time_Loading = np.sum(A[:, [2]], axis = 0);
    Time_Swinging = np.sum(A[:, [3]], axis = 0);
    Time_Grading_2 = np.sum(A[:, [4]], axis = 0);
    Time_Traveling = np.sum(A[:, [5]], axis = 0);

    Time_all = Time_Digging + Time_Grading_1 + Time_Loading + Time_Swinging + Time_Grading_2 + Time_Traveling;

    A_drop= np.delete(A, 6, axis=1) #axis = 1 means across columns; Here delete idling column

    df = pd.DataFrame(A)
    df.columns = ['Digging', 'Grading 1', 'Loading', 'Swinging', 'Grading 2', 'Traveling', 'Idling']
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm')
    plt.xticks(rotation=45)
    plt.show()


# sum_A_drop = np.sum(A_drop, axis=1); # Sum time of all activites per SOC change in rows
# mean_sum = np.mean(sum_A_drop)
# W = np.diag(sum_A_drop/mean_sum)

# b_ref = statistics.median(b)
# weight_soc = np.minimum(np.abs(b) / b_ref, 1).flatten()
# weight_soc = weight_soc / np.mean(weight_soc)
# W = np.diag(weight_soc)


W = np.eye(m) 

reg_param = 0e-4

"""""""""""""""""""""""""""""""""""""""CVXPY"""""""""""""""""""""""""""""""""

z, objective = solve_activity_power(A, b, n, W, reg_param, grading)

"""""""""""""""""""""""""""""""""""""""PRINTING"""""""""""""""""""""""""""""""""

if (grading == "False"):
    
    print(f"\nObjective with weighing and regularization {reg_param} is:  {objective:.5f}\n")

    print(f"\nDigging:   {z[0]:.2f} kW")
    print(f"\nLoading:   {z[1]:.2f} kW")
    print(f"\nSwinging:  {z[2]:.2f} kW")
    print(f"\nTraveling: {z[3]:.2f} kW")
    print(f"\nIdling:    {z[4]:.2f} kW")

    print(f"\n\nDigging:   {Time_Digging[0]*100/Time_all[0]:.0f}%")
    print(f"\nLoading:   {Time_Loading[0]*100/Time_all[0]:.0f}%")
    print(f"\nSwinging:  {Time_Swinging[0]*100/Time_all[0]:.0f}%")
    print(f"\nTraveling: {Time_Traveling[0]*100/Time_all[0]:.0f}%")
    
    
else:
    
    print(f"\nObjective with weighing and regularization {reg_param} is:  {objective:.5f}\n")

    print(f"\nDigging:   {z[0]:.2f} kW")
    print(f"\nGrading 1:   {z[1]:.2f} kW")
    print(f"\nLoading:   {z[2]:.2f} kW")
    print(f"\nSwinging:  {z[3]:.2f} kW")
    print(f"\nGrading 2:  {z[4]:.2f} kW")
    print(f"\nTraveling: {z[5]:.2f} kW")
    print(f"\nIdling:    {z[6]:.2f} kW")
    
    print(f"\n\nDigging:   {Time_Digging[0]*100/Time_all[0]:.0f}%")
    print(f"\n\nGrading 1:   {Time_Grading_1[0]*100/Time_all[0]:.0f}%")
    print(f"\nLoading:   {Time_Loading[0]*100/Time_all[0]:.0f}%")
    print(f"\nSwinging:  {Time_Swinging[0]*100/Time_all[0]:.0f}%")
    print(f"\n\nGrading 2:   {Time_Grading_2[0]*100/Time_all[0]:.0f}%")
    print(f"\nTraveling: {Time_Traveling[0]*100/Time_all[0]:.0f}%")

    



# Digging_power = z.value[0]*3600;
# Loading_power = z.value[1]*3600;
# Swinging_power = z.value[2]*3600;
# Traveling_power = z.value[3]*3600;
# Idling_power = z.value[4]*3600;

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

# labels = ['Digging', 'Loading', 'Swinging', 'Traveling', 'Idling']
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

# Keep figure windows open after the script finishes (only when launched from a terminal).
if sys.flags.interactive == 0 and plt.get_fignums():
    plt.ioff()
    plt.show()

