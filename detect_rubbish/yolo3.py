import cv2
import numpy as np
from IPython.display import display, Image
import os


def load_yolo3_model(img_path,config_file, weights_file):

    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(current_dir, config_file)
    weights_path = os.path.join(current_dir, weights_file)

    net = cv2.dnn.readNet(weights_path, config_path)

    # net = cv2.dnn.readNet("yolov3_training_last.weights", "yolov3_testing.cfg")
    classes = ["Organic", "Non Organic"]
    layer_names = net.getLayerNames()
    output_layers = [layer_names[i - 1] for i in net.getUnconnectedOutLayers()]
    colors = np.random.uniform(0, 255, size=(len(classes), 3))
    
    img = cv2.imread(img_path)
    img = cv2.resize(img, None, fx=0.1, fy=0.1)
    height, width, channels = img.shape
    
    blob = cv2.dnn.blobFromImage(img, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)
    
    class_ids = []
    confidences = []
    boxes = []
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)

                # Rectangle coordinates
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)

                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.4, 0.4)
    
    font = cv2.FONT_HERSHEY_PLAIN
    for i in range(len(boxes)):
        if i in indexes:
            x, y, w, h = boxes[i]
            label = str(classes[class_ids[i]])
            
            # new code
            confidence = str(round(confidences[i], 2))
            
            color = colors[class_ids[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, label, (x, y + 30), font, 3, color, 2)

            # new code
            cv2.putText(img, label , (x, y + 20), font, 2, (255, 255, 255), 2)

    return img

    
    
    