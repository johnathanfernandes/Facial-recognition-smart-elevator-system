#Image Processing and Pattern Recognition Course Project
#Smart Elevator using Facial Recognition
#Aneesh Poduval, Johnathan Fernandes, Sarthak Chudgar

#This program uses the generated database to detect faces and trigger the arduino demo elevator system

import cv2
import numpy as np
import serial 
import time
import winsound
import pickle
ser=serial.Serial('COM5',9600)
v=cv2.VideoCapture(0)
time.sleep(2)
fdc=cv2.CascadeClassifier(r'C:/python/python36/cascades/data/haarcascade_frontalface_alt2.xml')
recognizer = cv2.face.LBPHFaceRecognizer_create()
labels={"person_name":1}
with open("labels.pickle",'rb')as f:
    og_labels= pickle.load(f)
    labels = {v:k for k,v in og_labels.items()}
recognizer.read("trainner.yml")
#ser=serial.Serial('com4',9600)
frequency = 2500  # Set Frequency To 2500 Hertz
duration = 500  # Set Duration To 1000 ms == 1 second
while(1):
    #    serial.write('s')#Value of s is 0
        d,i=v.read()
        gray= cv2.cvtColor(i,cv2.COLOR_BGR2GRAY)
        #print(i.shape)
        fire=fdc.detectMultiScale(gray,1.3,9)
        print(fire)
        if(len(fire)>=1):
         for x,y,w,h in fire:
              #print(x,y,w,h)
              roi_gray = gray[y:y+h, x:x+w]
              id_,conf=recognizer.predict(roi_gray)
              if conf>=45 and conf<=85:
                      print(id_)
                      print(labels[id_])
                      if(id_==0):
                              time.sleep(0.5)
                              if(id_==0):
                                      ser.write(b'A')
                      elif(id_==1):
                              time.sleep(0.5)
                              if(id_==1):
                                      ser.write(b'S')
                      #print(conf)
              font=cv2.FONT_HERSHEY_SIMPLEX
              name=labels[id_]
              cv2.putText(i,name,(x,y),font,0.7,(255,255,255),2,cv2.LINE_AA)
              cv2.rectangle(i,(x,y),(x+w,y+h),(255,0,0),1)
   #           ser1.write(0)
              #time.sleep(0.2)
              #winsound.Beep(frequency, duration)
            # roi_gray = gray[y:y+h, x:x+w]
        cv2.imshow('image',i)
      #  serial.write('p')
       # if(len(fire)==1): 
        #   print('Face is detected')
        k=cv2.waitKey(5)
        if(k==ord('q')):
                cv2.destroyAllWindows()
                break