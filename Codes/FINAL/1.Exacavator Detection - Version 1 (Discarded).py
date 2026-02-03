import pandas as pd
import numpy as np
import xlwings as xw
import os
import re
from datetime import datetime, timedelta
import cv2
from tqdm import tqdm
import pathlib
import easyocr
import warnings

warnings.filterwarnings("ignore")

#%%%%%%%%%%%%%%%%%% Utility functions
def extract_date_from_path(video_path):
    match = re.search(r'([A-Za-z]{3,9})\s+(\d{1,2}),\s*(\d{4})', video_path)
    if match:
        month_str, day, year = match.groups()
        try:
            month_num = datetime.strptime(month_str, "%b").month
        except ValueError:
            month_num = datetime.strptime(month_str, "%B").month
        return f"{year}-{month_num:02d}-{int(day):02d}"
    return "unknown_date"

def is_valid_time(text):
    match = re.match(r'(\d{1,2}):(\d{2})', text)
    if match:
        hh, mm = int(match.group(1)), int(match.group(2))
        if 0 <= hh <= 24 and 0 <= mm < 60:
            return True
    return False

#%%%%%%%%%%%%%%%%%% Step 1: Extract frames only if empty
def extract_frames(video_path, fps_extract=1/60):
    video_name = pathlib.Path(video_path).stem
    date_str = extract_date_from_path(video_path)
    output_folder = fr"C:\Users\shubh\Desktop\Research work with AVIK\frames_{video_name}_{date_str}"
    os.makedirs(output_folder, exist_ok=True)

    existing_frames = [f for f in os.listdir(output_folder) if f.lower().endswith('.jpg')]
    if len(existing_frames) > 0:
        print(f"⚡ Skipping extraction for {video_name} — {len(existing_frames)} frames already exist.")
        return output_folder

    cap = cv2.VideoCapture(video_path)
    video_fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_interval = int(video_fps / fps_extract)
    saved_count = 0

    for i in tqdm(range(total_frames), desc=f"Extracting frames for {video_name}"):
        ret, frame = cap.read()
        if not ret:
            break
        if i % frame_interval == 0:
            frame_filename = os.path.join(output_folder, f"frame_{saved_count:06d}.jpg")
            cv2.imwrite(frame_filename, frame)
            saved_count += 1
    cap.release()
    print(f"✅ Extracted {saved_count} frames → '{output_folder}/'")
    return output_folder

#%%%%%%%%%%%%%%%%%% Step 2: Determine best OCR rotation/scale
def determine_best_ocr_params(sample_frames, reader):
    best_count = 0
    best_params = (0, 1.0)

    rotations = [0, 1, 2, 3]
    scales = [1.0, 0.75, 0.5]

    for r in rotations:
        for s in scales:
            valid_found = 0
            for frame_path in sample_frames:
                frame = cv2.imread(frame_path)
                if frame is None:
                    continue
                resized = cv2.resize(frame, None, fx=s, fy=s)
                rotated = np.rot90(resized, k=r)
                results = reader.readtext(rotated)
                for _, text, _ in results:
                    text = text.strip().replace('.', ':')
                    if is_valid_time(text):
                        valid_found += 1
                        break
            if valid_found > best_count:
                best_count = valid_found
                best_params = (r, s)
    print(f"🧠 Best OCR Params → rotation={best_params[0]*90}°, scale={best_params[1]:.2f}")
    return best_params

#%%%%%%%%%%%%%%%%%% Step 3: OCR all frames using learned params
def extract_timestamps_from_frames(frame_folder):
    reader = easyocr.Reader(['en'], gpu=False)
    frame_files = sorted([f for f in os.listdir(frame_folder) if f.lower().endswith('.jpg')])
    data = []
    prev_time = None
    failed_count = 0

    # Pick sample frames to learn rotation/scale
    sample_frames = [os.path.join(frame_folder, f) for f in frame_files[:min(10, len(frame_files))]]
    best_rotation, best_scale = determine_best_ocr_params(sample_frames, reader)

    for idx, frame_file in enumerate(tqdm(frame_files, desc=f"OCR on {os.path.basename(frame_folder)}")):
        frame_path = os.path.join(frame_folder, frame_file)
        frame = cv2.imread(frame_path)
        if frame is None:
            continue

        frame_resized = cv2.resize(frame, None, fx=best_scale, fy=best_scale)
        frame_rotated = np.rot90(frame_resized, k=best_rotation)
        results = reader.readtext(frame_rotated)

        timestamp_str = None
        for _, text, _ in results:
            text = text.strip().replace('.', ':')
            if is_valid_time(text):
                timestamp_str = text
                break

            if timestamp_str is None:
                failed_count += 1
                if prev_time is not None and is_valid_time(prev_time):
                    new_time = (datetime.strptime(prev_time, "%H:%M") + timedelta(minutes=1)).strftime("%H:%M")
                    timestamp_str = new_time
                else:
                    timestamp_str = "Extract Failed"


        prev_time = timestamp_str
        data.append({
            'frame_number': idx,
            'frame_file': frame_file,
            'timestamp': timestamp_str
        })

    print(timestamp_str)
    df = pd.DataFrame(data)
    print(f"🕒 Completed OCR for {os.path.basename(frame_folder)} | Failed: {failed_count}")
    return df

#%%%%%%%%%%%%%%%%%% Step 4: Save to Excel
def save_dataframes_to_excel(df_list, sheet_names, excel_path):
    wb = xw.Book()
    for df, sheet_name in zip(df_list, sheet_names):
        sht = wb.sheets.add(sheet_name[:31])
        sht.range('A1').value = df
    try:
        wb.sheets['Sheet1'].delete()
    except Exception:
        pass
    wb.save(excel_path)
    wb.close()
    print(f"📘 Saved all sheets → '{excel_path}'")

#%%%%%%%%%%%%%%%%%% Step 5: Main workflow
# video_paths = [
#     r"C:/Users/shubh/Desktop/Research work with AVIK/Video construction/Day 2-Oct 21, 2025/Excavator/videos/Day_2.mp4",
#     r"C:/Users/shubh/Desktop/Research work with AVIK/Video construction/Day 3-Oct 22, 2025/Excavator/videos/Day_3.mp4",
#     r"C:/Users/shubh/Desktop/Research work with AVIK/Video construction/Day 4-Oct 23, 2025/Excavator/videos/Day_4.mp4",
# ]
video_paths = [r"C:/Users/shubh/Desktop/Research work with AVIK/Video construction/Day 4-Oct 23, 2025/Excavator/videos/Day_4.mp4"]
excel_path = r"C:/Users/shubh/Desktop/Research work with AVIK/minute_by_minute_activities_all.xlsx"

all_dfs = []
sheet_names = []

for video_path in video_paths:
    frames_folder = extract_frames(video_path, fps_extract=1/60)
    df = extract_timestamps_from_frames(frames_folder)
    all_dfs.append(df)
    date_str = extract_date_from_path(video_path)
    sheet_names.append(date_str if date_str != "unknown_date" else pathlib.Path(video_path).stem)

save_dataframes_to_excel(all_dfs, sheet_names, excel_path)