from ultralytics import YOLO
import torch
import os

# ------------------- Check GPU -------------------
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Using device: {device}")

# ------------------- Dataset -------------------
# Set the path to your Roboflow YOLOv8 downloaded data
data_yaml = r"C:\Users\shubh\Desktop\Research work with AVIK\Test\Object detection\data.yaml"

# ------------------- Model -------------------
# Use pretrained YOLOv8n for transfer learning
model = YOLO(r"C:\Users\shubh\Desktop\Research work with AVIK\yolov8n.pt")  # pretrained YOLOv8n
model.to(device)

# ------------------- Training -------------------
results = model.train(
    data=data_yaml,
    epochs=50,
    imgsz=480,
    batch=16,
    device=device,
    name="yolo_excavator_custom",   # folder name for this training run
    project=r"C:\Users\shubh\Desktop\Research work with AVIK\Test"      # where weights are saved
)

# After training, the best model is saved in:
best_model_path = os.path.join(r"C:\Users\shubh\Desktop\Research work with AVIK\Test",
                               "yolo_excavator_custom",
                               "weights",
                               "best.pt")
print(f"✅ Training complete! Best model saved at: {best_model_path}")
