#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 13 04:18:30 2026

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
from datetime import datetime, timedelta                           #pip install dill --user
import cvxpy as cp
from scipy.optimize import nnls
import re

start_time = time.time()

"""""""""""""""""""""""""""""""""""""""GROUND TRUTH"""""""""""""""""""""""""""""""""

Data_actual = pd.read_excel (r"C:\Users\shubh\Desktop\New folder\Ideal.xlsx", sheet_name="Day3")

t = Data_actual["Time"].astype(str).str.strip()

def to_seconds(x: str) -> int:
    parts = [int(p) for p in x.split(":")]
    if len(parts) == 2:      # mm:ss
        m, s = parts
        return 60*m + s
    elif len(parts) == 3:    # hh:mm:ss
        h, m, s = parts
        return 3600*h + 60*m + s
    else:
        raise ValueError(f"Bad time format: {x!r}")

# Detect range rows
is_range = t.str.contains(r"\s*-\s*", regex=True)

# Extract start/end in seconds
rng = t[is_range].str.extract(r"^\s*([0-9:]+)\s*-\s*([0-9:]+)\s*$")
Data_actual.loc[is_range, "start_sec"] = rng[0].map(to_seconds)
Data_actual.loc[is_range, "end_sec"]   = rng[1].map(to_seconds)

# Convert to Int64 nullable type
Data_actual["start_sec"] = Data_actual.get("start_sec").astype("Int64")
Data_actual["end_sec"]   = Data_actual.get("end_sec").astype("Int64")

# -------------------
# APPLY OFFSET HERE
# -------------------
offset_sec = to_seconds("55:58")   # = 3358 sec
offset_frame = offset_sec * 60     # = 201480 frames (assuming 60FPS)

# Convert to frames (minus offset)
Data_actual["start_frame"] = Data_actual["start_sec"]*60 - offset_frame
Data_actual["end_frame"]   = Data_actual["end_sec"]*60   - offset_frame

chunks = []
for i in range(len(Data_actual)):
    
    start = Data_actual["start_frame"].iloc[i]
    end = Data_actual["end_frame"].iloc[i]
    activity = Data_actual["Activity"].iloc[i]

    idx = range(start, end)     # exclusive end

    tmp = pd.DataFrame({"Frame": idx, "Actual_activity": activity})
    chunks.append(tmp)

df_actual = pd.concat(chunks, ignore_index=True).sort_values("Frame").reset_index(drop=True)
df_actual["Frame"]=df_actual["Frame"]+1

df_actual["Predicted_activity"]=np.nan

    
"""""""""""""""""""""""""""""""""""""""PREDICTED ACTIVITY"""""""""""""""""""""""""""""""""

Data_prediction = pd.read_csv (r"C:\Users\shubh\Desktop\New folder\Activity_Output_2_18.csv");
Data_prediction.loc[Data_prediction["activity_name"].eq("dumping"), "activity_name"] = "swinging"


for i in range(len(Data_prediction)):
    frame = Data_prediction["frame"].iloc[i];
    predcited_activity = Data_prediction["activity_name"].iloc[i];
    
    df_actual.loc[df_actual["Frame"].eq(frame), "Predicted_activity"] = predcited_activity
    
df_actual = df_actual.dropna()

"""""""""""""""""""""""""""""""""""""""COMPARISON"""""""""""""""""""""""""""""""""


df_actual["match"] = (df_actual["Actual_activity"].astype(str).str.strip().str.casefold().eq(df_actual["Predicted_activity"].astype(str).str.strip().str.casefold())).astype(int)
matches = df_actual["match"].sum();

Digging_actual = len(df_actual.loc[df_actual["Actual_activity"] == "Digging"]);
Loading_actual = len(df_actual.loc[df_actual["Actual_activity"] == "Loading"]);
Swinging_actual = len(df_actual.loc[df_actual["Actual_activity"] == "Swinging"]);
Travelling_actual = len(df_actual.loc[df_actual["Actual_activity"] == "Travelling"]);
Idling_actual = len(df_actual.loc[df_actual["Actual_activity"] == "Idling"]);

Digging_pred = len(df_actual.loc[df_actual["Predicted_activity"] == "digging"]);
Loading_pred = len(df_actual.loc[df_actual["Predicted_activity"] == "loading"]);
Swinging_pred = len(df_actual.loc[df_actual["Predicted_activity"] == "swinging"]);
Travelling_pred = len(df_actual.loc[df_actual["Predicted_activity"] == "travelling"]);
Idling_pred = len(df_actual.loc[df_actual["Predicted_activity"] == "idling"]);

print(f"\nPoint-to-point match: {(matches*100/len(df_actual)):.0f}"+"%")


if(Digging_actual!=0):
    print(f"\nActual/Predicted Digging: {Digging_actual:.0f}"+" /"+f"{Digging_pred:.0f}"+" ~ "+f" {(Digging_pred-Digging_actual)*100/Digging_actual:.1f}")
else:
    print(f"\nActual/Predicted Digging: {Digging_actual:.0f}"+" /"+f"{Digging_pred:.0f}")

if(Loading_actual!=0):
    print(f"\nActual/Predicted Loading: {Loading_actual:.0f}"+" /"+f"{Loading_pred:.0f}" + " ~"+f" {(Loading_pred-Loading_actual)*100/Loading_actual:.1f}")
else:
    print(f"\nActual/Predicted Loading: {Loading_actual:.0f}"+" /"+f"{Loading_pred:.0f}")

if(Swinging_actual!=0):
    print(f"\nActual/Predicted Swinging: {Swinging_actual:.0f}"+" /"+f"{Swinging_pred:.0f}" + " ~"+f" {(Swinging_pred-Swinging_actual)*100/Swinging_actual:.1f}")
else:
    print(f"\nActual/Predicted Swinging: {Swinging_actual:.0f}"+" /"+f"{Swinging_pred:.0f}")

if(Travelling_actual!=0):
    print(f"\nActual/Predicted Traveling: {Travelling_actual:.0f}"+" /"+f"{Travelling_pred:.0f}"+ " ~"+f" {(Travelling_pred-Travelling_actual)*100/Travelling_actual:.1f}")
else:
    print(f"\nActual/Predicted Traveling: {Travelling_actual:.0f}"+" /"+f"{Travelling_pred:.0f}")

if(Idling_actual!=0):
    print(f"\nActual/Predicted Idling: {Idling_actual:.0f}"+" /"+f"{Idling_pred:.0f}"+ " ~"+f" {(Idling_pred-Idling_actual)*100/Idling_actual:.1f}")
else:
    print(f"\nActual/Predicted Idling: {Idling_actual:.0f}"+" /"+f"{Idling_pred:.0f}")






