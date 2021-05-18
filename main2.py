import cv2
import dlib
import numpy as np


org=cv2.imread("contents/face2.jpeg")
gray_pic=cv2.cvtColor(org,cv2.COLOR_BGR2GRAY)

detector=dlib.get_frontal_face_detector()
predicator=dlib.shape_predictor("contents/shape_predictor_68_face_landmarks.dat")

faceInPic=detector(gray_pic)
# cv2.imshow("pic",gray_pic)
# cv2.waitKey(0) 

print('faceInPic')
print(faceInPic)

for face in faceInPic:
    
    print(face)
    x1=face.left()
    y1=face.top()
    x2=face.right()
    y2=face.bottom()
    cv2.rectangle(gray_pic, (x1,y1),(x2,y2), (0,0,255),3)
    
    landmark=predicator(gray_pic,face)

    for n in range(0,68):
            x=landmark.part(n).x
            y=landmark.part(n).y
            # print(x,y)
            cv2.circle(gray_pic,(x,y),4,(255,0,0),-1)
    

cv2.imshow("pic",gray_pic)
cv2.waitKey(0) 

