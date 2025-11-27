import os
import csv
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt

# ===============================
# USER CONFIG
# ===============================
TRACK_CSV = r"C:\Users\shubh\Desktop\Test\Track_Output.csv"
OUTPUT_CSV = r"C:\Users\shubh\Desktop\Test\Idling_segments.csv"

FPS = 59
WINDOW = 40                    # per paper
DIST_THRESHOLD = 0.2            # px
AREA_THRESHOLD_PERCENT = 0.5    # % of mean area

# Post-processing params
MERGE_GAP_SEC = 1.0             # merge segments if gap < 1 sec
MIN_IDLE_DURATION_SEC = 3.0     # drop idles shorter than 3 sec

# Medium smoothing params
SAVGOL_WINDOW = 11              # must be odd
SAVGOL_POLYORDER = 2
ROLL_MEDIAN_WINDOW = 5

# Optional: save plots to folder or set to None
PLOT_DIR = r"C:\Users\shubh\Desktop\Test"
# ===============================


# ---------------------------------------------------------
# Utility: Convert seconds → HH:MM:SS.mmm
# ---------------------------------------------------------
def seconds_to_hms(seconds):
    hrs = int(seconds // 3600)
    mins = int((seconds % 3600) // 60)
    secs = seconds % 60
    return f"{hrs:02d}:{mins:02d}:{secs:06.3f}"


# ---------------------------------------------------------
# Smoothing function (medium)
# ---------------------------------------------------------
def smooth_signal_med(signal):
    n = len(signal)
    if n < 3:
        return signal.copy()

    # Ensure savgol window validity
    w = SAVGOL_WINDOW
    if w >= n:
        w = n - 1 if (n - 1) % 2 == 1 else n - 2
    if w < 3:
        w = 3

    sg = savgol_filter(signal, w, SAVGOL_POLYORDER, mode="interp")

    # Rolling median for spike suppression
    s = pd.Series(sg)
    s = s.rolling(window=min(ROLL_MEDIAN_WINDOW, n),
                  center=True,
                  min_periods=1).median()

    return s.values


# ---------------------------------------------------------
# Idling detection for a single track
# ---------------------------------------------------------
def compute_idling_for_track(track_df):
    if track_df.empty or len(track_df) < 2:
        return []

    track_df = track_df.sort_values("frame").reset_index(drop=True)
    frames = track_df["frame"].values.astype(int)
    x1, y1 = track_df["x1"].values, track_df["y1"].values
    x2, y2 = track_df["x2"].values, track_df["y2"].values

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = (x2 - x1) * (y2 - y1)

    # Smooth signals
    cx_s = smooth_signal_med(cx)
    cy_s = smooth_signal_med(cy)
    area_s = smooth_signal_med(area)

    area_mean = np.mean(area_s)
    area_std_thresh = (AREA_THRESHOLD_PERCENT / 100.0) * area_mean

    n = len(frames)
    idle_mask = np.zeros(n, dtype=bool)

    # Sliding window detection
    if n >= WINDOW:
        for i in range(0, n - WINDOW + 1):
            cxw = cx_s[i:i+WINDOW]
            cyw = cy_s[i:i+WINDOW]
            aw = area_s[i:i+WINDOW]

            dist = np.sqrt(np.diff(cxw)**2 + np.diff(cyw)**2)
            dA = np.abs(np.diff(aw))

            if np.std(dist) < DIST_THRESHOLD and np.std(dA) < area_std_thresh:
                idle_mask[i:i+WINDOW] = True
    else:
        # Short tracks → evaluate entire track as one window
        dist = np.sqrt(np.diff(cx_s)**2 + np.diff(cy_s)**2)
        dA = np.abs(np.diff(area_s))
        if np.std(dist) < DIST_THRESHOLD and np.std(dA) < area_std_thresh:
            idle_mask[:] = True

    # Convert mask → raw segments
    raw_segments = []
    in_seg = False
    for i, val in enumerate(idle_mask):
        if val and not in_seg:
            in_seg = True
            s = frames[i]
        elif not val and in_seg:
            in_seg = False
            e = frames[i-1]
            raw_segments.append((s, e))
    if in_seg:
        raw_segments.append((s, frames[-1]))

    # -------------------
    # Post-processing
    # -------------------
    merged = []
    if raw_segments:
        gap_f = int(MERGE_GAP_SEC * FPS)
        min_f = int(MIN_IDLE_DURATION_SEC * FPS)

        cs, ce = raw_segments[0]
        for s, e in raw_segments[1:]:
            if s - ce <= gap_f:
                ce = e
            else:
                if (ce - cs + 1) >= min_f:
                    merged.append((cs, ce))
                cs, ce = s, e

        if (ce - cs + 1) >= min_f:
            merged.append((cs, ce))

    return merged


# ---------------------------------------------------------
# Visualization function
# ---------------------------------------------------------
def visualize_track(track_df, segments, savepath=None):
    if track_df.empty:
        return

    track_df = track_df.sort_values("frame").reset_index(drop=True)
    frames = track_df["frame"].values

    x1, y1 = track_df["x1"].values, track_df["y1"].values
    x2, y2 = track_df["x2"].values, track_df["y2"].values

    cx = (x1 + x2) / 2.0
    cy = (y1 + y2) / 2.0
    area = (x2 - x1) * (y2 - y1)

    cx_s = smooth_signal_med(cx)
    cy_s = smooth_signal_med(cy)
    area_s = smooth_signal_med(area)

    mov_raw = np.concatenate([[0], np.sqrt(np.diff(cx)**2 + np.diff(cy)**2)])
    mov_s = np.concatenate([[0], np.sqrt(np.diff(cx_s)**2 + np.diff(cy_s)**2)])

    id_mask = np.zeros(len(frames))
    for (s, e) in segments:
        id_mask[(frames >= s) & (frames <= e)] = 1

    fig, axs = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    axs[0].plot(frames, mov_raw, label="raw movement")
    axs[0].plot(frames, mov_s, label="smoothed movement")
    axs[0].legend()
    axs[0].grid(True)

    axs[1].plot(frames, area, label="area raw")
    axs[1].plot(frames, area_s, label="area smoothed")
    axs[1].legend()
    axs[1].grid(True)

    axs[2].plot(frames, id_mask, drawstyle="steps-mid")
    axs[2].set_ylabel("Idle (0/1)")
    axs[2].grid(True)

    plt.tight_layout()

    if savepath:
        plt.savefig(savepath)
        plt.close()
    else:
        plt.show()


# ---------------------------------------------------------
# Main
# ---------------------------------------------------------
def main():
    df = pd.read_csv(TRACK_CSV)
    output_rows = []

    if PLOT_DIR:
        os.makedirs(PLOT_DIR, exist_ok=True)

    for track_id in sorted(df["track_id"].unique()):
        tdf = df[df["track_id"] == track_id]

        segments = compute_idling_for_track(tdf)

        for (s, e) in segments:
            duration = (e - s + 1) / FPS

            start_sec = s / FPS
            end_sec = e / FPS

            start_hms = seconds_to_hms(start_sec)
            end_hms = seconds_to_hms(end_sec)

            output_rows.append([
                track_id,
                s,
                e,
                round(duration, 3),
                round(start_sec, 3),
                round(end_sec, 3),
                start_hms,
                end_hms
            ])

        # Save plots if enabled
        if PLOT_DIR:
            plot_path = os.path.join(PLOT_DIR, f"track_{track_id}.png")
            visualize_track(tdf, segments, savepath=plot_path)

    # Write output CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "track_id",
            "start_frame",
            "end_frame",
            "duration_sec",
            "start_time_sec",
            "end_time_sec",
            "start_time_hms",
            "end_time_hms"
        ])
        writer.writerows(output_rows)

    print("Idling detection DONE.")
    print(f"Saved CSV → {OUTPUT_CSV}")
    if PLOT_DIR:
        print(f"Saved plots → {PLOT_DIR}")


if __name__ == "__main__":
    main()
