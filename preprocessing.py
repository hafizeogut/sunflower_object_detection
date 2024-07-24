#Veri setini bölmek için yazılan bir koddur.
from sklearn.model_selection import train_test_split
import cv2
import os
import yaml

#veri kümesi kök dizini 
root_dir = "datasets/sunflower" # bu dizin altında resimler ve etiket dosyaları bulunur.
valid_formats = [".png","jpg","jpeg",".txt"] #veri kümesinde yer almasına izin verilen formatlar belirleniyor.

def file_paths(root,valid_formats):şdfkfd