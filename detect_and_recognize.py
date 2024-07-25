from ultralytics import YOLO
from easyocr import Reader
import time
import torch
import cv2 
import os
import csv

def detect_number_plates(image,model,display = False):
    start = time.time()
    detections = model.predict(image)[0].boxes.data
    print(detections)

model = YOLO("/home/hafizeogut/Desktop/sunflower_object_detection/runs/detect/yolov8n_sunflower/weights/best.pt")
image = "/home/hafizeogut/Desktop/5s3xv00kn7.jpg"



detect_number_plates(image,model)