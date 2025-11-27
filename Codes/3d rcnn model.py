import torch
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from torchvision import transforms
import cv2

model = fasterrcnn_resnet50_fpn(pretrained=True).eval()
transform = transforms.Compose([transforms.ToTensor()])

# Path to your video
video_path = r"C:\Users\shubh\Desktop\Test\Day_2 - Trim.mp4"
cap = cv2.VideoCapture(video_path)

# Get video details
fps = int(cap.get(cv2.CAP_PROP_FPS))
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Output video path
out_path = r"C:\Users\shubh\Desktop\Test\Day_2 - RCNN.mp4"
out = cv2.VideoWriter(out_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (width, height))

frame_count = 0
processed = 0

print("Starting detection...")
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1
    if frame_count % 10 != 0:
        continue
    img_tensor = transform(frame)
    preds = model([img_tensor])[0]
    boxes = preds['boxes']
    scores = preds['scores']
    labels = preds['labels']
    for i, score in enumerate(scores):
        if score > 0.8:  # confidence threshold
            box = boxes[i].detach().numpy().astype(int)
            cv2.rectangle(frame, (box[0], box[1]), (box[2], box[3]), (0,255,0), 2)
    out.write(frame)
    processed += 1
    if processed % 10 == 0:
        print(f"Processed {processed} frames (skipped {frame_count - processed})...")
cap.release()
out.release()