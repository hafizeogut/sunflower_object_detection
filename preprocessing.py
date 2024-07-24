#Veri setini bölmek için yazılan bir koddur.
from sklearn.model_selection import train_test_split
import cv2
import os
import yaml

#veri kümesi kök dizini 
root_dir = "../sunflower_object_detection" # bu dizin altında resimler ve etiket dosyaları bulunur.
valid_formats = [".png","jpg","jpeg",".txt"] #veri kümesinde yer almasına izin verilen formatlar belirleniyor.

def file_paths(root, valid_formats):
    file_paths = []
    # Dizin yolunun var olup olmadığını kontrol edin
    if not os.path.exists(root):
        print(f"Verilen dizin mevcut değil: {root}")
        return
    
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            #extension: dosya adının uzantısı olarak ayarlandı
            extension = os.path.splitext(filename)[1].lower()

            if extension in valid_formats:
                #dirpath ve filename değerlerini birleştirerek tam dosya yolunu oluşturur.
                file_path = os.path.join(dirpath,filename)
                file_paths.append(file_path)

    return file_paths
 