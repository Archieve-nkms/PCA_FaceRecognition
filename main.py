import os
import cv2
import matplotlib.pyplot as plt

def load_images_from_folder(path):
    result = []
    for file_name in os.listdir(path):
        img = cv2.imread(os.path.join(path, file_name))
        if img is not None:
            result.append(img)
    return result


face_images = load_images_from_folder("C:\\Users\\nkm\\Desktop\\PCA\\Images")
print(len(face_images), " image files loaded")