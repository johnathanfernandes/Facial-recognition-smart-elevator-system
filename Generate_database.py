#Image Processing and Pattern Recognition Course Project
#Smart Elevator using Facial Recognition
#Aneesh Poduval, Johnathan Fernandes, Sarthak Chudgar

#This program uses the training images to generate the database

import cv2
import os
import numpy as np
from PIL import Image
import pickle
BASE_DIR=os.path.dirname(os.path.abspath(__file__))
image_dir=os.path.join(BASE_DIR,"known")
fdc=cv2.CascadeClassifier(r'C:/python/python36/cascades/data/haarcascade_frontalface_alt2.xml')
#recognizer= cv2.face.LBPHFaceRecognizer_create()
#recognizer = cv2.face.createLBPHFaceRecognizer()
recognizer = cv2.face.LBPHFaceRecognizer_create()
current_id=0
label_ids={}
y_labels=[]
x_train=[]
for root, dirs, files in os.walk("known"):
    for file in files:
        if file.endswith("png")or file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path))
            #print(label, path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            #print(label_ids)
            pil_img=Image.open(path).convert("L")
            #size= (550,550)
            #final_image=pil_img.resize(size,Image.ANTIALIAS)
            img_array=np.array(pil_img,"uint8")
            #print(img_array)
            fire=fdc.detectMultiScale(img_array,1.3,9)
            for x,y,w,h in fire:
                roi= img_array[y:y+h, x:x+w]
                x_train.append(roi)
                y_labels.append(id_)
#print(y_labels)
#print(x_train)
with open("labels.pickle",'wb')as f:
    pickle.dump(label_ids,f)
recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainner.yml")