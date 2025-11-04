# If not installed, run these commands in your terminal:
# pip install ultralytics torch torchvision opencv-python

from ultralytics import YOLO

# Load YOLOv12 model (nano, small, medium, large, or extra large)
model = YOLO('yolo12n.pt')  # Use yolov12s.pt, yolov12m.pt, or yolov12l.pt for larger models

# Train the model
results = model.train(
    data="/home/lora-laptop/AI/rnd/dataset/data.yaml",  # data.yaml location with train/val/test splits
    imgsz=640,                 # image size (adjust as needed)
    epochs=100,                 # number of training epochs
    batch=4,                  # batch size (adjust according to GPU RAM)
    device='0',                # '0' ensures use of first GPU (change to '-1' for CPU)
    name='yolov12_vehicle_custom',  # experiment/run name
    patience=5,             # early stopping patience (optional)
)

# Validate on the test set
val_results = model.val(
    data='/home/lora-laptop/AI/rnd/dataset/data.yaml',
    split='test',              # evaluate using the test split
    device='0'                 # ensure validation also runs on GPU
)

# After training, check the runs/detect/yolov8_vehicle_custom/weights directory
# "best.pt" contains your best model weights (highest mAP)
