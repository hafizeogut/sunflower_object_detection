from ultralytics import YOLO
from easyocr import Reader
import time
import torch
import cv2 
import os
import csv

confidence_threshold =0.4
color = (255,255,0)


#ayçiçek tespit fonksiyonu
def detect_sunflower(image,model,display = False):
    start = time.time()#işlemin başlangıç zamanını al
    detections = model.predict(image)[0].boxes.data #Model ile tahmin yap ve tespit edilen kutuları al
    print(detections) #Tespit edilen kutuları yazdır
    print("shape",detections.shape)#Tespit edilen kutuları yazdır

    #Eğer tespit edilen kutular boş değilse
    if detections.shape != torch.Size([0,6]):
        boxes = [] #kutuları saklamak için boş bir kutu oluştur
        confidences = [] #güven değerlerini toplamak için boş bir liste oluştur
        for detection in detections:
            confidence = detection[4]
            if float(confidence) < confidence_threshold:
                continue
            boxes.append(detection[:4])
            confidences.append(detection[4])
        print(f"{len(boxes)} Number sunflower have been detected")
        sunflower_list = []

        print(boxes)
        for i in range (len(boxes)):
            xmin, ymin, xmax, ymax = int(boxes[i][0]),int(boxes[i][1]),int(boxes[i][2]),int(boxes[i][3])
            sunflower_list.append([xmin, ymin, xmax, ymax])
            cv2.rectangle(image,(xmin,ymin),color,2)
            text = "Sunflower: {:.2f}%".format(confidences[i]* 100,cv2.FONT_HERSHEY_SIMPLEX,0.5,color,2)
            cv2.putText(image,text,(xmin,ymin -5))
        if display:
            sunflower = image[ymin:ymax, xmin:xmax]
            cv2.imshow(f"Number plate",sunflower)
        end = time.time()
        print(f"Time to detect the sunflower: {end-start}*1000: .0f milliseconds")

        return sunflower_list

    else:
        print("No number plate have been detected")
        return [ ]
    
if __name__ == "__main__"
    #YOLO modelini yükle
    model = YOLO("/home/hafizeogut/Desktop/sunflower_object_detection/runs/detect/yolov8n_sunflower/weights/best.pt")

    #Test görüntüsünün yolu
    image = "/home/hafizeogut/Desktop/sunflower_object_detection/sunflower/images/test/00af3df40b_jpg.rf.4296f31f894c60a792281b03f25c6145.jpg"

    detect_sunflower(image,model)
"""
print(detections)
         x1         ,y1         ,x2         ,y2        ,confidence:Model nesneyi tanımlma güven aralığı  ,class:Tespit edilen nesnenin sınıf
tensor([[1.6039e+02, 0.0000e+00, 5.5177e+02, 5.8120e+02, 9.6662e-01, 1.5000e+01],
        [5.6618e-02, 0.0000e+00, 1.0329e+02, 5.9409e+02, 5.9874e-01, 1.5000e+01]], device='cuda:0')
shape torch.Size([2, 6])

print("shape",detections.shape)
shape torch.Size([2, 6])

Tensorun şekli, 2 tespit ve her tespit için 6 değer olduğunu gösterir.
"""

