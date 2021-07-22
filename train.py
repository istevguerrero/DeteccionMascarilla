import cv2
import os
import numpy as np

dataPath = "C:/Users/imste/Desktop/Reconocimiento Facial/Dataset_faces"

dir_list = os.listdir(dataPath)

labels = []

facesData = []

label = 0

for name_dir in dir_list:

    dir_path = dataPath + "/" + name_dir

    for file_name in os.listdir(dir_path):

        image_path = dir_path + "/" + file_name

        image = cv2.imread(image_path, 0)

        facesData.append(image)
        
        labels.append(label)

    label += 1

face_mask = cv2.face.LBPHFaceRecognizer_create()

print("Entrenando...")

face_mask.train(facesData, np.array(labels))

face_mask.write("face_mask_model.xml")

print("Modelo Almacenado")