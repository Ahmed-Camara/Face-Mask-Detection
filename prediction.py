import cv2
from keras.preprocessing import image 
import numpy as np

from keras.models import load_model
model = load_model('model/face_mask_model.h5')
face_cascade = cv2.CascadeClassifier('resources/haarcascade_frontalface_default.xml')

labels = ['Mask','Non Mask']
cap = cv2.VideoCapture(0)
cap.set(3,600)
cap.set(4,580)

while True:

    success, img = cap.read()

    if success == False:
        print('could not read video')
        exit(1)


    faces = face_cascade.detectMultiScale(img)

    for(x,y,w,h) in faces:

        

        roi_img = img[y:y+h, x:x+w]
        rerect_sized=cv2.resize(roi_img,(224,224))
        rerect_sized = np.array(rerect_sized)
        reshaped=np.reshape(rerect_sized,(1,224,224,3))
        result=model.predict(reshaped)
        label = labels[int((result>0.5)*1)]

        cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,255),2)    
        cv2.putText(img, label, (x, y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)
        
    cv2.imshow('Live Mask Detection',img)

    if cv2.waitKey(1) & 0XFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()