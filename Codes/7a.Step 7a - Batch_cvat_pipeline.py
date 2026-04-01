import os
import requests
from tqdm import tqdm
import time

# -----------------------------
# CONFIGURATION
# -----------------------------

CVAT_URL = "http://localhost:8080"
USERNAME = "ShubhanMital"
PASSWORD = "Viratkohli@55"

#VIDEOS_FOLDER = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Video construction\Clipped Videos from other days"
VIDEOS_FOLDER = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\CVAT_Stuff\cvat_data\videos"
RAW_FOLDER = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Video construction\Clipped Videos from other days"
CVAT_XML_FOLDER = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Video construction\Clipped Videos from other days\cvat_annotations"
EXPORT_FOLDER = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\Video construction\Clipped Videos from other days\exported_annotations"

os.makedirs(CVAT_XML_FOLDER, exist_ok=True)
os.makedirs(EXPORT_FOLDER, exist_ok=True)

# -----------------------------
# LOGIN TO CVAT
# -----------------------------
session = requests.Session()

print("Waiting for CVAT server...")

while True:
    try:
        r = requests.get("http://localhost:8080")
        if r.status_code == 200:
            break
    except:
        pass
    time.sleep(5)

print("CVAT server ready")

login_data = {
    "username": USERNAME,
    "password": PASSWORD
}

login = session.post(
    f"{CVAT_URL}/api/auth/login",
    json=login_data
)

print("Login status:", login.status_code)

if login.status_code not in [200, 201]:
    raise Exception("Login failed")

print("Logged into CVAT")

# Extract the auth token and CSRF cookie, then attach them to the session
token_key = login.json().get("key")
csrf_token = session.cookies.get("csrftoken")

session.headers.update({
    "Authorization": f"Token {token_key}",
    "X-CSRFToken": csrf_token
})

# -----------------------------
# STEP 1 — CREATE TASKS
# -----------------------------
session.timeout = 600
tasks = {}

for video in os.listdir(VIDEOS_FOLDER):
    print("Files in folder:")
    print(os.listdir(VIDEOS_FOLDER))

    if not video.endswith(".mp4"):
        continue

    video_path = os.path.join(VIDEOS_FOLDER, video)
    print(os.path.getsize(video_path) / (1024**3), "GB")

    task_name = video.replace(".mp4","")

    print("Creating task:", task_name)

    # files = {"client_files[0]": open(video_path,"rb")}

    data = {
        "name": task_name,
        "labels":[
            {"name": "Digging", "color": "#1a6ff6"},
            {"name": "Travelling", "color": "#0ea2c3"},
            {"name": "Idling", "color": "#f41ab4"},
            {"name": "Swinging", "color": "#4b5920"},
            {"name": "Loading", "color": "#af2568"},
            {"name": "Dumping", "color": "#22b8de"}
        ],
        "overlap": 0
    }
    
    r = session.post(f"{CVAT_URL}/api/tasks", json=data, timeout=60)
    print("Status:", r.status_code)

    if r.status_code != 201:
        print("Task creation failed:", r.text)
        continue
    
    task_id = r.json()["id"]
    data_upload = {
        "server_files": [video],
        "image_quality": 100
    }
    
    r = session.post(
        f"{CVAT_URL}/api/tasks/{task_id}/data",
        json=data_upload
    )
    
    print("Upload status:", r.status_code)
    
    tasks[task_name] = task_id

    print("Tasks created")
    print("Waiting for CVAT to finish frame extraction...")
# Automatically poll until CVAT finishes frame extraction
for task_id in tasks.values():
    while True:
        r = session.get(f"{CVAT_URL}/api/tasks/{task_id}")
        
        # Check if CVAT has sliced the video into jobs yet
        jobs_count = r.json().get("jobs", {}).get("count", 0)
        
        if jobs_count > 0:
            print(f"✅ Task {task_id} frame extraction complete!")
            break
            
        print(f"Waiting for frame extraction on Task {task_id}...")
        time.sleep(15)

# -----------------------------
# STEP 2 — CONVERT RAW XML
# -----------------------------

def convert_raw_to_cvat(raw_xml, output_xml):

    # placeholder converter
    # replace with your excel_to_cvat logic

    with open(raw_xml,"r") as f:
        data = f.read()

    with open(output_xml,"w") as f:
        f.write(data)

for file in os.listdir(RAW_FOLDER):

    if not file.endswith("_raw.xml"):
        continue

    raw_path = os.path.join(RAW_FOLDER,file)

    out_name = file.replace("_raw.xml",".xml")

    out_path = os.path.join(CVAT_XML_FOLDER,out_name)

    convert_raw_to_cvat(raw_path,out_path)

print("Raw XML converted")

# -----------------------------
# STEP 3 — UPLOAD ANNOTATIONS
# -----------------------------

for xml in os.listdir(CVAT_XML_FOLDER):

    task_name = xml.replace(".xml","")

    task_id = tasks.get(task_name)

    if task_id is None:
        continue

    print("Uploading:", xml)

    files = {"annotation_file":open(os.path.join(CVAT_XML_FOLDER,xml),"rb")}

    r = session.post(
        f"{CVAT_URL}/api/tasks/{task_id}/annotations",
        files=files,
        data={"format":"CVAT for Video 1.1"}
    )

print("Annotations uploaded")

# -----------------------------
# STEP 4 — EXPORT DATASETS
# -----------------------------


print("Starting Step 4: Exporting Annotations...")

for name, task_id in tasks.items():
    print(f"Preparing export: {name} (Task {task_id})")

    # 1. Use the working /dataset/export endpoint with lowercase "v" and save_images=False
    r = session.post(
        f"{CVAT_URL}/api/tasks/{task_id}/dataset/export",
        params={
            "format": "CVAT for video 1.1", # STRICTLY case-sensitive!
            "save_images": False            # This ensures we only download annotations, not gigabytes of frames
        }
    )

    if r.status_code not in [201, 202]:
        print(f"❌ Failed to start export for {name}. Status: {r.status_code}, Response: {r.text}")
        continue

    rq_id = r.json().get("rq_id")
    
    # 2. Poll the specific request ID status
    while True:
        r_req = session.get(f"{CVAT_URL}/api/requests/{rq_id}")
        req_data = r_req.json()
        
        # CVAT uses 'state' to track progress ('queued', 'started', 'finished', 'failed')
        state = req_data.get("status", "").lower()

        if state == "finished":
            result_url = req_data.get("result_url")
            
            # 3. Download the prepared file using the provided result_url
            print(f"Export ready on server! Downloading {name}...")
            
            # Handle whether CVAT returns an absolute or relative URL
            download_url = result_url if result_url.startswith("http") else f"{CVAT_URL}{result_url}"
            
            r_dl = session.get(download_url)
            
            with open(os.path.join(EXPORT_FOLDER, f"{name}.zip"), "wb") as f:
                f.write(r_dl.content)
                
            print(f"✅ Export {name}.zip downloaded successfully!")
            break
            
        elif state in ["failed", "error"]:
            print(f"❌ Export failed for {name}. Reason: {req_data.get('message')}")
            break
            
        else:
            print(f"Server is building the export (Status: {state})... waiting 5 seconds.")
            time.sleep(5)

print("All exports complete!")

#%%
import os
import cv2
import pandas as pd

# -----------------------------
# CONFIGURATION
# -----------------------------
VIDEOS_FOLDER = r"C:\Users\shubh\Desktop\Research_work_with_AVIK\CVAT_Stuff\cvat_data\videos"

def get_video_stats(folder_path):
    data = []
    
    print(f"Scanning folder: {folder_path}...\n")
    
    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".mp4"):
            continue
            
        filepath = os.path.join(folder_path, filename)
        
        # Open the video file to read its metadata
        video = cv2.VideoCapture(filepath)
        
        if not video.isOpened():
            print(f"Error opening video: {filename}")
            continue
            
        # Extract properties
        fps = video.get(cv2.CAP_PROP_FPS)
        frame_count = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Calculate duration in seconds and format as MM:SS
        if fps > 0:
            duration_sec = frame_count / fps
            minutes = int(duration_sec // 60)
            seconds = int(duration_sec % 60)
            duration_str = f"{minutes}m {seconds}s"
        else:
            duration_str = "0m 0s"
            
        # Release the video file lock
        video.release()
        
        # Add to our table data
        data.append({
            "Video Name": filename,
            "FPS": round(fps, 2),
            "Total Frames": frame_count,
            "Duration": duration_str
        })
        
    # Create and display the table
    if data:
        df = pd.DataFrame(data)
        # Sort alphabetically by video name
        df = df.sort_values(by="Video Name").reset_index(drop=True)
        
        print("-" * 65)
        print(df.to_string(index=False))
        print("-" * 65)
    else:
        print("No .mp4 files found in the specified directory.")

# Run the function
get_video_stats(VIDEOS_FOLDER)