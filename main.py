import cv2
import dlib
# import numpy as np


cap=cv2.VideoCapture(0)

detector=dlib.get_frontal_face_detector()
predicator=dlib.shape_predictor("contents/shape_predictor_68_face_landmarks.dat")

while True:
    
    _,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    
    faces=detector(gray)
    
    for face in faces:
        
        print(face)
        x1=face.left()
        y1=face.top()
        x2=face.right()
        y2=face.bottom()
        # cv2.rectangle(frame, (x1,y1),(x2,y2), (0,0,255),3)
        landmark=predicator(gray,face)
        # print(landmark)
        for n in range(0,68):
            x=landmark.part(n).x
            y=landmark.part(n).y
            # print(x,y)
            cv2.circle(frame,(x,y),4,(255,0,0),-1)
        
    cv2.imshow("Frame",frame)
     
    key=cv2.waitKey(1)
    
    if key == 27:
        break




