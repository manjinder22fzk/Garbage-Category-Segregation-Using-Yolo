from django.shortcuts import render
from django.conf import settings
from django.core.files.storage import FileSystemStorage
from .yolo3 import load_yolo3_model
from .yolo7 import load_yolo7_model
from .yolo8 import load_yolo8_model
import cv2
import os
import io
import sys
import numpy as np
from PIL import Image
from torchvision import transforms
from django.conf import settings
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt

def home(request):
    return render(request, 'detect_rubbish/home.html')

def detect_objects(request):
    if request.method == 'POST' and request.FILES['image']:
        uploaded_file = request.FILES['image']
        fs = FileSystemStorage()
        filename = fs.save(uploaded_file.name, uploaded_file)
        uploaded_image_url = fs.url(filename)
        image_path = os.path.join(settings.MEDIA_ROOT, filename)
        image = cv2.imread(image_path)
        
        if request.POST.get('yolo_v3'):
            config_file = 'yolov3_testing.cfg'
            weights_file = 'yolov3_training_last.weights'
            processed_image = load_yolo3_model(image_path, config_file, weights_file)
            
        elif request.POST.get('yolo_v7'):
            processed_image = load_yolo7_model(image_path)
        
        elif request.POST.get('yolo_v8'):
            processed_image = load_yolo8_model(image_path)

        result_image_filename = 'processed_' + filename
        result_image_path = os.path.join(settings.MEDIA_ROOT, result_image_filename)
        cv2.imwrite(result_image_path, processed_image)
        result_image_url = fs.url(result_image_filename)
        
        return render(request, 'detect_rubbish/home.html', {'result_image': result_image_url})
    
    return render(request,'detect_rubbish/home.html')

