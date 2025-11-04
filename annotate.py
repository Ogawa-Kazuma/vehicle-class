import os
import cv2
import numpy as np
import torch
from ultralytics import YOLO

# --- CONFIGURATION ---
# IMPORTANT: Update these paths and parameters for your specific project
MODEL_PATH = 'rehh.pt' # e.g., 'yolov8n.pt'
NEW_IMAGES_DIR = '/home/lora-laptop/Downloads/Thai_vehicle_classification_dataset_TR1/images'
OUTPUT_LABELS_DIR = '/home/lora-laptop/Downloads/Thai_vehicle_classification_dataset_TR1/labels'
# You must map your model's output class IDs to your dataset's class names
CLASS_NAMES = ['Class 1', 'Class 2', 'Class 3', 'Class 4', 'Class 5', 'Class 6'] 
CONFIDENCE_THRESHOLD = 0.5 # Minimum confidence for a prediction to be saved

def load_trained_model(MODEL_PATH):
    """
    Loads a custom trained Ultralytics YOLO model from a .pt file.
    The model is automatically configured for inference.
    """
    # model_path will be your defined MODEL_PATH variable (e.g., '/path/to/rehh.pt')
    print(f"Loading custom YOLOv12 model from: {MODEL_PATH}...")
    
    # CRITICAL: Pass the path to your custom .pt file directly to the YOLO constructor
    model = YOLO(MODEL_PATH) 
    
    # The Ultralytics YOLO object handles loading the architecture and weights
    # and is automatically in evaluation mode for inference.
    return model

def model_inference(model, image_path, confidence_threshold):
    """
    Runs YOLOv12 inference and returns detections in ABSOLUTE PIXEL format.
    
    Output format: [ [x_min, y_min, x_max, y_max, class_id, confidence], ... ]
    """
    # 1. Run inference using the model's predict method
    # Note: Ultralytics handles the image loading internally.
    # We set 'conf' to the threshold to filter low-confidence boxes immediately.
    # Set 'imgsz' to the size you trained with (e.g., 640).
    results = model.predict(
        source=image_path, 
        conf=confidence_threshold, 
        verbose=False, # Suppress the frequent logging output for cleaner script runs
        imgsz=640,
        device='0' # Use '0' or 'cuda' if you have a GPU, otherwise 'cpu'
    )
    
    # 2. Extract and format detections
    final_detections = []
    
    # Check if the result list is not empty and has boxes data
    if results and len(results) > 0:
        res = results[0] # Get the results object for the first (and only) image
        
        # Check if any detections were found and extract the tensor data
        if res.boxes is not None:
            # res.boxes.data is a tensor containing: 
            # [x_min, y_min, x_max, y_max, confidence, class_id]
            # Coordinates (xyxy) are already in ABSOLUTE PIXELS
            
            # Convert the PyTorch tensor to a NumPy array for easy iteration/handling
            data = res.boxes.data.cpu().numpy() 
            
            for detection in data:
                x_min, y_min, x_max, y_max, conf, cls = detection
                
                # Format: [x_min, y_min, x_max, y_max, class_id, confidence]
                # We cast class_id (cls) to integer
                final_detections.append([
                    x_min, y_min, x_max, y_max, int(cls), conf
                ])

    return final_detections

def convert_to_yolo_format(detections, img_width, img_height):
    """
    Converts absolute pixel coordinates [x_min, y_min, x_max, y_max, class_id, conf] 
    to normalized YOLO format: [class_id x_center y_center box_w box_h]
    """
    yolo_labels = []
    
    for x_min, y_min, x_max, y_max, class_id, _ in detections:
        # Calculate center coordinates and width/height
        x_center = (x_min + x_max) / 2
        y_center = (y_min + y_max) / 2
        box_w = x_max - x_min
        box_h = y_max - y_min
        
        # Normalize coordinates (YOLO format requires 0.0 to 1.0)
        x_center_norm = x_center / img_width
        y_center_norm = y_center / img_height
        box_w_norm = box_w / img_width
        box_h_norm = box_h / img_height
        
        # Format the line: CLASS_ID X_CENTER Y_CENTER WIDTH HEIGHT
        label_line = f"{int(class_id)} {x_center_norm:.6f} {y_center_norm:.6f} {box_w_norm:.6f} {box_h_norm:.6f}"
        yolo_labels.append(label_line)
        
    return yolo_labels

def main():
    # Setup directories
    if not os.path.exists(OUTPUT_LABELS_DIR):
        os.makedirs(OUTPUT_LABELS_DIR)

    # 1. Load the trained model
    model = load_trained_model(MODEL_PATH)

    # 2. Iterate through new images
    for image_name in os.listdir(NEW_IMAGES_DIR):
        if image_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            image_path = os.path.join(NEW_IMAGES_DIR, image_name)
            
            # 3. Run inference
            # This returns a list of [x_min, y_min, x_max, y_max, class_id, conf]
            detections = model_inference(model, image_path, CONFIDENCE_THRESHOLD)
            
            if not detections:
                print(f"No detections above threshold for {image_name}.")
                continue

            # Load image again to get dimensions for normalization
            img = cv2.imread(image_path)
            if img is None: continue 
            img_height, img_width = img.shape[:2]

            # 4. Convert predictions to annotation format (e.g., YOLO)
            yolo_annotations = convert_to_yolo_format(detections, img_width, img_height)

            # 5. Save annotations
            # Label file name must match the image name but with a .txt extension
            label_name = os.path.splitext(image_name)[0] + '.txt'
            label_path = os.path.join(OUTPUT_LABELS_DIR, label_name)
            
            with open(label_path, 'w') as f:
                f.write('\n'.join(yolo_annotations))
            
            print(f"Annotated and saved: {label_name} with {len(yolo_annotations)} objects.")

if __name__ == "__main__":
    main()