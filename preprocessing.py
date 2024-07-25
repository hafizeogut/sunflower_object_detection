from sklearn.model_selection import train_test_split
import cv2
import os
import yaml

# Veri kümesi kök dizini
root_dir = "../sunflower_object_detection"  # Bu dizin altında resimler ve etiket dosyaları bulunur.
valid_formats = [".png", ".jpg", ".jpeg", ".txt"]  # Veri kümesinde yer almasına izin verilen formatlar belirleniyor.

def file_paths(root, valid_formats):
    file_paths = []
    # Dizin yolunun var olup olmadığını kontrol ediliyor
    if not os.path.exists(root):
        print(f"Verilen dizin mevcut değil: {root}")
        return []
    # Dizin içindeki dosyalar dolaşılıyor
    for dirpath, dirnames, filenames in os.walk(root):
        for filename in filenames:
            # Dosya uzantısını al ve küçük harfe çevir
            extension = os.path.splitext(filename)[1].lower()

            # Geçerli formatlardan biriyle eşleşiyorsa:
            if extension in valid_formats:
                # dirpath ve filename değerlerini birleştirerek tam dosya yolunu oluştur
                file_path = os.path.join(dirpath, filename)
                file_paths.append(file_path)

    return file_paths

image_paths = file_paths(root_dir + "/images", valid_formats[:3])
label_paths = file_paths(root_dir + "/labels", [valid_formats[-1]])

# Veri setinin %20 doğrulama ve %80 eğitim kümesi olarak ayarlanıyor
X_train, X_val_test, y_train, y_val_test = train_test_split(image_paths, label_paths, test_size=0.2, random_state=42)
# X_train ve y_train: Eğitim seti. image_paths ve label_paths veri setlerinin %80'ini içerir.
# X_val_test ve y_val_test: Doğrulama ve test seti olarak kullanılacak olan kalan %20'lik kısım.

X_valid, X_test, y_valid, y_test = train_test_split(X_val_test, y_val_test, test_size=0.7, random_state=42)
# X_valid ve y_valid: Doğrulama seti. X_val_test ve y_val_test veri setlerinin %30'unu içerir.
# X_test ve y_test: Test seti. X_val_test ve y_val_test veri setlerinin %70'ini içerir.

def write_to_file(image_dir, label_dir, X):
    os.makedirs(image_dir, exist_ok=True)
    os.makedirs(label_dir, exist_ok=True)
    for image_path in X:
        #Benim dosya formatım :76-995e92405b6b68b40b_jpg.rf.8ff7e6863350068ffc68b02bf0f4def3.jpg
        #8ff7e6863350068ffc68b02bf0f4def3.jpg olsaydı img_name = image_path.split("/")[-1].split(".")[0] yeterli olacaktı.
        img_name = image_path.split("/")[-1].split(".")[0] + "."+image_path.split("/")[-1].split(".")[1] +"."+image_path.split("/")[-1].split(".")[2] # Dosya adını al
        
        img_ext = image_path.split("/")[-1].split(".")[-1]  # Dosya uzantısını al

        image = cv2.imread(image_path)
        cv2.imwrite(f"{image_dir}/{img_name}.{img_ext}", image)

        with open(f"{label_dir}/{img_name}.txt", "w") as f:
            with open(f"{root_dir}/labels/{img_name}.txt", "r") as label_file:
                f.write(label_file.read())

write_to_file("sunflower/images/train", "sunflower/labels/train", X_train)
write_to_file("sunflower/images/valid", "sunflower/labels/valid", X_valid)
write_to_file("sunflower/images/test", "sunflower/labels/test", X_test)


# data = {
#     "path": "../sunflower_object_detection/sunflower",
#     "train":"images/train",
#     "val":"images/valid",
#     "test":"images/test",

#     "names":["sunflower"]
# }

# with open("sunflower.yaml","w") as file:
#     yaml.dump(data,file)