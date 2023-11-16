import cv2
import numpy as np
from PIL import Image
import os
import yolov7
import torch
from torchvision.ops import nms

def load_yolo7_model(img_path):
    model = yolov7.load(r'C:\Users\Money Brar\Desktop\Garbage_Detection\detect_rubbish\yolov7_best_weights.pt')
    model.conf = 0.25  # NMS confidence threshold
    model.iou = 0.45  # NMS IoU threshold
    model.classes = None  # (optional list) filter by class
    
    img = img_path
    results = model(img)
    predictions = results.pred[0]
    boxes = predictions[:, :4] # x1, y1, x2, y2
    scores = predictions[:, 4]
    categories = predictions[:, 5]
    img = cv2.imread(img)
    for i in range(len(boxes)):
        x1, y1, x2, y2 = boxes[i]
        label = "Organic" if categories[i] == 0 else "Non Organic"
        cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 5)
        color = (0, 0, 255) if categories[i] == 0 else (255, 0, 0)
        cv2.putText(img, label, (int(x1), int(y1-10)), cv2.FONT_HERSHEY_SIMPLEX, 3, color, 5, cv2.LINE_AA)

    return img
