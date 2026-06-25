import sys
import os
from pathlib import Path

print("="*60)
print("PRE-FLIGHT CHECK: LIBRARIES & FILES")
print("="*60)

# -----------------------------------------------------------
# 1. CHECK LIBRARIES (IMPORTS)
# -----------------------------------------------------------
print("\n[1] Checking Python Libraries...")

required_libraries = [
    ("pandas", "pd"),
    ("cv2", "cv2 (opencv-python)"),
    ("torch", "torch"),
    ("torchvision", "torchvision"),
    ("numpy", "numpy"),
    ("scipy", "scipy"),
    ("matplotlib", "matplotlib"),
    ("tqdm", "tqdm"),
    ("ultralytics", "ultralytics (YOLO)"),
    ("openpyxl", "openpyxl (Excel support)") 
]

all_libs_ok = True

for lib_name, display_name in required_libraries:
    try:
        __import__(lib_name)
        print(f"  ✅ {display_name:<25} : INSTALLED")
    except ImportError:
        print(f"  ❌ {display_name:<25} : MISSING (Run: pip install {lib_name})")
        all_libs_ok = False

# -----------------------------------------------------------
# 2. CHECK CUDA / GPU
# -----------------------------------------------------------
print("\n[2] Checking Compute Device...")

try:
    import torch
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  ✅ CUDA Available           : YES ({gpu_name})")
        print(f"  ✅ PyTorch Version          : {torch.__version__}")
    else:
        print("  ⚠️ CUDA Available           : NO (Code will run on CPU - VERY SLOW)")
        print(f"  ⚠️ PyTorch Version          : {torch.__version__}")
except:
    print("  ❌ PyTorch not installed/working.")

# -----------------------------------------------------------
# 3. CHECK FILE PATHS
# -----------------------------------------------------------
print("\n[3] Checking Project Files...")

# Paths from your configuration
MODEL_PATH      = Path(r"/mnt/nvme1/avik_shubhan/resnet3d/resnet3d_best_kinetics_2.pth")
VIDEO_PATH      = Path(r"/mnt/nvme1/avik_shubhan/resnet3d/Day_3.mp4")
YOLO_PATH       = Path(r"/mnt/nvme1/avik_shubhan/resnet3d/best.pt")
TASKS_XLSX      = Path(r"/mnt/nvme1/avik_shubhan/resnet3d/Tasks.xlsx")
OUTPUT_BASE_DIR = Path(r"/mnt/nvme1/avik_shubhan/resnet3d_1/optimization_runs")
MEAN_PATH       = Path("/mnt/nvme1/avik_shubhan/resnet3d/dataset_mean.npy")
STD_PATH        = Path("/mnt/nvme1/avik_shubhan/resnet3d/dataset_std.npy")

files_to_check = [
    ("ResNet Model", MODEL_PATH),
    ("Video File", VIDEO_PATH),
    ("YOLO Model", YOLO_PATH),
    ("Tasks Excel", TASKS_XLSX),
    ("Mean Stats", MEAN_PATH),
    ("Std Stats", STD_PATH)
]

all_files_ok = True

for name, path in files_to_check:
    if path.exists():
        print(f"  ✅ {name:<20} : FOUND")
    else:
        print(f"  ❌ {name:<20} : MISSING -> {path}")
        all_files_ok = False

# -----------------------------------------------------------
# 4. CHECK WRITE PERMISSIONS
# -----------------------------------------------------------
print("\n[4] Checking Write Permissions...")
try:
    OUTPUT_BASE_DIR.mkdir(parents=True, exist_ok=True)
    test_file = OUTPUT_BASE_DIR / "test_write.tmp"
    test_file.touch()
    test_file.unlink()
    print(f"  ✅ Output Directory       : WRITABLE ({OUTPUT_BASE_DIR})")
except Exception as e:
    print(f"  ❌ Output Directory       : NOT WRITABLE ({OUTPUT_BASE_DIR})")
    print(f"     Error: {e}")
    all_files_ok = False

# -----------------------------------------------------------
# SUMMARY
# -----------------------------------------------------------
print("\n" + "="*60)
if all_libs_ok and all_files_ok:
    print("🚀 SYSTEM READY. You can run the optimization code now.")
else:
    print("🛑 SYSTEM NOT READY. Please fix the ❌ items above.")
print("="*60)