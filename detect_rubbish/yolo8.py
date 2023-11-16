# # pip install ultralytics -- run this  command for installing yolov8 


import cv2
import numpy as np
from PIL import Image
import os
import torch
from torchvision.ops import nms
from ultralytics import YOLO as yolov8

def load_yolo8_model(img_path):
    
    model = yolov8(r'C:\Users\Money Brar\Desktop\Garbage_Detection\detect_rubbish\yolov8_best_weights.pt')
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.classes = None  # (optional list) filter by class

    img = cv2.imread(img_path)
    results = model.predict(img_path)
    results_obj = results[0]
    
    # Number of detected objects
    num_detections = len(results_obj.boxes.xyxy)

    # Detected object bounding boxes and classes
    for i in range(num_detections):
        box_xyxy = results_obj.boxes.xyxy[i]
        box_conf = results_obj.boxes.conf[i]
        class_id = int(results_obj.boxes.cls[i])
        class_name = results_obj.names[class_id]
        x1, y1, x2, y2 = [int(coord) for coord in box_xyxy]

        print(f"Object {i + 1}: {class_name}")
        print(f"Bounding box: x1={x1}, y1={y1}, x2={x2}, y2={y2}")
        print(f"Confidence: {box_conf}\n")

        # Draw bounding box on the image
        img = cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 5)

        # Draw class name and confidence on the image
        color = (0, 0, 255) if class_id == 0 else (255, 0, 0)
        label = f"{class_name}: {box_conf:.2f}"
        img = cv2.putText(img, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 4)

    return img
