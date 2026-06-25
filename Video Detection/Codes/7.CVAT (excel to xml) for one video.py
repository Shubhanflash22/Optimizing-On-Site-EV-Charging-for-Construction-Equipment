import pandas as pd
import re
import json
from lxml import etree

# =====================================
# CONFIG
# =====================================
DEFAULT_ACTIVITY = "Idling"
FPS = float(input("Enter video FPS (e.g., 59.94): "))
TOTAL_FRAMES = int(input("Enter total number of frames (e.g., 199060): "))
VIDEO_WIDTH = int(input("Enter video width (e.g., 1280): "))
VIDEO_HEIGHT = int(input("Enter video height (e.g., 720): "))
EXCEL_PATH = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Tasks.xlsx"   
XML_OUT = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\output_cvat.xml"

# Bounding box = entire frame
BBOX_XTL = "0"
BBOX_YTL = "0"
BBOX_XBR = str(VIDEO_WIDTH)
BBOX_YBR = str(VIDEO_HEIGHT)

# =====================================
# Display video info
# =====================================
DURATION_SEC = TOTAL_FRAMES / FPS

print(f"\nVideo Info:")
print(f"  FPS: {FPS:.2f}")
print(f"  Total Frames: {TOTAL_FRAMES}")
print(f"  Duration: {DURATION_SEC:.2f} seconds ({DURATION_SEC/60:.2f} minutes)")
print()

# =====================================
# Parse "MM:SS-MM:SS" automatically
# =====================================
def parse_time_range(time_str):
    # Remove all spaces
    t = time_str.replace(" ", "")
    
    # Split by '-'
    start_str, end_str = t.split("-")
    
    # Convert to seconds
    def to_sec(x):
        mm, ss = map(int, x.split(":"))
        return mm * 60 + ss
    
    return to_sec(start_str), to_sec(end_str)

# =====================================
# Load Excel
# =====================================
df = pd.read_excel(EXCEL_PATH)

# Parse times
start_seconds = []
end_seconds = []

for t in df["Time"]:
    s, e = parse_time_range(str(t))
    start_seconds.append(s)
    end_seconds.append(e)

df["start_sec"] = start_seconds
df["end_sec"] = end_seconds

# Sort by start time
df = df.sort_values("start_sec").reset_index(drop=True)

# =====================================
# Build timeline & fill missing gaps
# =====================================
segments = []

# If video starts before first annotation:
if df.loc[0, "start_sec"] > 0:
    segments.append({
        "activity": DEFAULT_ACTIVITY,
        "start_sec": 0,
        "end_sec": df.loc[0, "start_sec"]
    })

# Main + gaps
for i in range(len(df)):
    row = df.loc[i]
    
    # Add actual interval
    segments.append({
        "activity": row["Activity"],
        "start_sec": row["start_sec"],
        "end_sec": row["end_sec"]
    })
    
    # Add idling gap if needed
    if i < len(df) - 1:
        next_start = df.loc[i+1, "start_sec"]
        if row["end_sec"] < next_start:
            segments.append({
                "activity": DEFAULT_ACTIVITY,
                "start_sec": row["end_sec"],
                "end_sec": next_start
            })

# Add final segment to cover remaining video
last_end_sec = df.loc[len(df)-1, "end_sec"]
video_duration_sec = TOTAL_FRAMES / FPS
if last_end_sec < video_duration_sec:
    segments.append({
        "activity": DEFAULT_ACTIVITY,
        "start_sec": last_end_sec,
        "end_sec": video_duration_sec
    })

# =====================================
# Convert to frames
# =====================================
prev_end_frame = None

for seg in segments:
    start_frame = int(seg["start_sec"] * FPS)
    end_frame = int(seg["end_sec"] * FPS)

    # shift start_frame to avoid overlap
    if prev_end_frame is not None and start_frame <= prev_end_frame:
        start_frame = prev_end_frame + 1

    seg["start_frame"] = start_frame
    seg["end_frame"] = end_frame

    prev_end_frame = end_frame

# Ensure last segment ends exactly at video end
if segments[-1]["end_frame"] < TOTAL_FRAMES - 1:
    segments[-1]["end_frame"] = TOTAL_FRAMES - 1

# =====================================
# EXPORT → CVAT XML (Tracks Format)
# =====================================
root = etree.Element("annotations")

# Version
version = etree.SubElement(root, "version")
version.text = "1.1"

# Meta section
meta = etree.SubElement(root, "meta")
job = etree.SubElement(meta, "job")

# Labels with proper structure
labels_elem = etree.SubElement(job, "labels")

# Define colors for each label
label_colors = {
    "Digging": "#1a6ff6",
    "Travelling": "#0ea2c3",
    "Idling": "#f41ab4",
    "Swinging": "#4b5920",
    "Loading": "#af2568",
    "Dumping": "#22b8de"
}

all_labels = sorted(set(seg["activity"] for seg in segments))
for label_text in all_labels:
    label = etree.SubElement(labels_elem, "label")
    
    name = etree.SubElement(label, "name")
    name.text = label_text
    
    color = etree.SubElement(label, "color")
    color.text = label_colors.get(label_text, "#ff0000")
    
    label_type = etree.SubElement(label, "type")
    label_type.text = "any"
    
    etree.SubElement(label, "attributes")

# Create tracks (one per segment)
track_id = 0
for seg in segments:
    track = etree.SubElement(root, "track")
    track.set("id", str(track_id))
    track.set("label", seg["activity"])
    track.set("source", "manual")
    
    # Start frame box (keyframe)
    box_start = etree.SubElement(track, "box")
    box_start.set("frame", str(seg["start_frame"]))
    box_start.set("keyframe", "1")
    box_start.set("outside", "0")
    box_start.set("occluded", "0")
    box_start.set("xtl", BBOX_XTL)
    box_start.set("ytl", BBOX_YTL)
    box_start.set("xbr", BBOX_XBR)
    box_start.set("ybr", BBOX_YBR)
    box_start.set("z_order", "0")
    box_start.text = " "
    
    # End frame box (keyframe with outside=1 to end the track)
    box_end = etree.SubElement(track, "box")
    box_end.set("frame", str(seg["end_frame"]))
    box_end.set("keyframe", "1")
    box_end.set("outside", "1")  # This marks the end of the track
    box_end.set("occluded", "0")
    box_end.set("xtl", BBOX_XTL)
    box_end.set("ytl", BBOX_YTL)
    box_end.set("xbr", BBOX_XBR)
    box_end.set("ybr", BBOX_YBR)
    box_end.set("z_order", "0")
    box_end.text = " "
    
    track_id += 1

# Save XML
tree = etree.ElementTree(root)
tree.write(XML_OUT, pretty_print=True, xml_declaration=True, encoding="utf-8")

print("DONE ✔")
print(f"Generated: {XML_OUT}")
print(f"\nTotal segments: {len(segments)}")
print(f"Total tracks created: {track_id}")
print(f"\nBounding box coordinates used:")
print(f"  Top-left: ({BBOX_XTL}, {BBOX_YTL})")
print(f"  Bottom-right: ({BBOX_XBR}, {BBOX_YBR})")
print(f"\nYou can adjust the bounding box coordinates in the script if needed.")