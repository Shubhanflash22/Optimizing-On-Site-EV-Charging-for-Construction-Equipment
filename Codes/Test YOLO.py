from ultralytics import YOLO
import cv2
import csv

# model = YOLO("yolov8n.pt")
model = YOLO(r"C:\Users\shubh\Desktop\Research work with AVIK\Test\yolo_excavator_custom\weights\best.pt")

# Path to your video
video_path = r"C:\Users\shubh\Desktop\Test\Test.mp4"
cap = cv2.VideoCapture(video_path)

# Get video details
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video path
out_path = r"C:\Users\shubh\Desktop\Test\Test-YOLO.mp4"
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))
csv_path = r"C:\Users\shubh\Desktop\Test\Test.csv"

# Create CSV file
csv_file = open(csv_path, "w", newline="")
writer = csv.writer(csv_file)
writer.writerow(["frame", "x1", "y1", "x2", "y2", "confidence", "class"])

frame_count = 0
processed = 0

# Applying the yolo model and saving the csv and video
print("Starting detection...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 1 != 0:
        continue
    
    results = model(frame, imgsz=480, verbose=False)
    
    if len(results[0].boxes) > 0:
        best_box = max(results[0].boxes, key=lambda b: float(b.conf[0]))
        x1, y1, x2, y2 = best_box.xyxy[0].tolist()
        conf = float(best_box.conf[0])
        cls = int(best_box.cls[0])
        annotated = frame.copy()
        cv2.rectangle(annotated, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
        cv2.putText(annotated, f"{cls}:{conf:.2f}", (int(x1), int(y1)-10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        out.write(annotated)
        writer.writerow([frame_count, int(x1), int(y1), int(x2), int(y2), conf, cls])
    else:
        out.write(frame)
    
    processed += 1
    if processed % 10 == 0:
        print(f"Processed {processed} frames (skipped {frame_count - processed})...")

cap.release()
out.release()
csv_file.close()