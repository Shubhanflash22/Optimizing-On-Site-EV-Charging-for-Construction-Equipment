import pandas as pd
import cv2
import os
from lxml import etree

# =====================================
# PATHS
# =====================================

EXCEL_PATH = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Video construction\Clipped Videos from other days\Labelling.xlsx"

VIDEO_FOLDER = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\CVAT_Stuff\cvat_data\videos"

DEFAULT_ACTIVITY = "Idling"


# =====================================
# VIDEO INFO FUNCTION
# =====================================

def get_video_info(video_path):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise Exception(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    cap.release()

    return fps, total_frames, width, height


# =====================================
# TIME PARSER
# =====================================

def parse_time_range(time_str):

    t = time_str.replace(" ", "")
    start_str, end_str = t.split("-")

    def to_sec(x):
        mm, ss = map(int, x.split(":"))
        return mm * 60 + ss

    return to_sec(start_str), to_sec(end_str)


# =====================================
# PROCESS ONE SHEET
# =====================================

def process_sheet(sheet_name, df):

    video_path = os.path.join(VIDEO_FOLDER, f"{sheet_name}.mp4")

    if not os.path.exists(video_path):
        print(f"Video not found for sheet: {sheet_name}")
        return

    print(f"\nProcessing {sheet_name}")

    FPS, TOTAL_FRAMES, VIDEO_WIDTH, VIDEO_HEIGHT = get_video_info(video_path)

    BBOX_XTL = "0"
    BBOX_YTL = "0"
    BBOX_XBR = str(VIDEO_WIDTH)
    BBOX_YBR = str(VIDEO_HEIGHT)

    # ---------------------------------
    # Parse time
    # ---------------------------------

    start_seconds = []
    end_seconds = []

    for i, t in enumerate(df["Time"]):
        try:
            s, e = parse_time_range(str(t))
            start_seconds.append(s)
            end_seconds.append(e)
        except Exception as err:
            print("\nERROR FOUND")
            print("Sheet:", sheet_name)
            print("Row:", i)
            print("Value in Time column:", t)
            raise err

    df["start_sec"] = start_seconds
    df["end_sec"] = end_seconds

    df = df.sort_values("start_sec").reset_index(drop=True)

    # ---------------------------------
    # Build segments
    # ---------------------------------

    segments = []

    if df.loc[0, "start_sec"] > 0:
        segments.append({
            "activity": DEFAULT_ACTIVITY,
            "start_sec": 0,
            "end_sec": df.loc[0, "start_sec"]
        })

    for i in range(len(df)):

        row = df.loc[i]

        segments.append({
            "activity": row["Activity"],
            "start_sec": row["start_sec"],
            "end_sec": row["end_sec"]
        })

        if i < len(df) - 1:

            next_start = df.loc[i+1, "start_sec"]

            if row["end_sec"] < next_start:
                segments.append({
                    "activity": DEFAULT_ACTIVITY,
                    "start_sec": row["end_sec"],
                    "end_sec": next_start
                })

    video_duration = TOTAL_FRAMES / FPS

    last_end = df.loc[len(df)-1, "end_sec"]

    if last_end < video_duration:

        segments.append({
            "activity": DEFAULT_ACTIVITY,
            "start_sec": last_end,
            "end_sec": video_duration
        })

    # ---------------------------------
    # Convert to frames
    # ---------------------------------

    prev_end_frame = None
    MAX_VALID_FRAME = TOTAL_FRAMES - 1

    for seg in segments:
        start_frame = int(seg["start_sec"] * FPS)
        end_frame = int(seg["end_sec"] * FPS)

        if prev_end_frame is not None and start_frame <= prev_end_frame:
            start_frame = prev_end_frame + 1

        # CAP the frames so they never exceed the video length
        seg["start_frame"] = min(start_frame, MAX_VALID_FRAME)
        seg["end_frame"] = min(end_frame, MAX_VALID_FRAME)

        prev_end_frame = seg["end_frame"]

    # Final safety check for the very last segment
    if segments[-1]["end_frame"] > MAX_VALID_FRAME:
        segments[-1]["end_frame"] = MAX_VALID_FRAME

    # ---------------------------------
    # BUILD XML (WITH META BLOCK)
    # ---------------------------------
    root = etree.Element("annotations")
    
    # CVAT works best when it sees the version and meta info
    version = etree.SubElement(root, "version")
    version.text = "1.1"

    meta = etree.SubElement(root, "meta")
    task = etree.SubElement(meta, "task")
    etree.SubElement(task, "size").text = str(TOTAL_FRAMES)
    
    # Add labels info so CVAT knows what these activities are
    labels = etree.SubElement(task, "labels")
    unique_activities = df["Activity"].unique().tolist()
    if DEFAULT_ACTIVITY not in unique_activities:
        unique_activities.append(DEFAULT_ACTIVITY)
        
    for act in unique_activities:
        label_node = etree.SubElement(labels, "label")
        etree.SubElement(label_node, "name").text = str(act)

    # ---------------------------------
    # TRACKS
    # ---------------------------------
    track_id = 0
    for seg in segments:
        # Skip segments that were squashed to 0 length by the capping logic
        if seg["start_frame"] >= seg["end_frame"] and seg != segments[0]:
            continue

        track = etree.SubElement(root, "track")
        track.set("id", str(track_id))
        track.set("label", str(seg["activity"]))
        track.set("source", "manual")

        # Start of activity
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

        # End of activity (outside=1 tells CVAT the track stops here)
        box_end = etree.SubElement(track, "box")
        box_end.set("frame", str(seg["end_frame"]))
        box_end.set("keyframe", "1")
        box_end.set("outside", "1")
        box_end.set("occluded", "0")
        box_end.set("xtl", BBOX_XTL)
        box_end.set("ytl", BBOX_YTL)
        box_end.set("xbr", BBOX_XBR)
        box_end.set("ybr", BBOX_YBR)
        box_end.set("z_order", "0")

        track_id += 1

    # ---------------------------------
    # SAVE XML
    # ---------------------------------

    xml_out = os.path.join(VIDEO_FOLDER, f"{sheet_name}_raw.xml")

    tree = etree.ElementTree(root)
    tree.write(xml_out, pretty_print=True, xml_declaration=True, encoding="utf-8")

    print(f"Saved: {xml_out}")


# =====================================
# MAIN LOOP
# =====================================

excel = pd.ExcelFile(EXCEL_PATH)

for sheet in excel.sheet_names:

    df = pd.read_excel(EXCEL_PATH, sheet_name=sheet)

    process_sheet(sheet, df)