import cv2 as cv,  face_recognition
import numpy as np

image_tusher=face_recognition.load_image_file('animesh_tusher/ani5.jpg')
image_tusher=cv.cvtColor(image_tusher,cv.COLOR_BGR2RGB)
image_test=face_recognition.load_image_file('animesh_tusher/ani6.jpg')
image_test=cv.cvtColor(image_test,cv.COLOR_BGR2RGB)

faceLoc=face_recognition.face_locations(image_tusher)[0] #we are giving only one image , that's why 0
encode_tusher=face_recognition.face_encodings(image_tusher)[0]
# print(faceLoc) it gives 4 values. top, right, bottom , left
cv.rectangle(image_tusher,(faceLoc[0],faceLoc[3]),(faceLoc[1],faceLoc[2]),(0,0,255),2)

#Detect face for test image
faceLocTest=face_recognition.face_locations(image_test)[0]
encode_test=face_recognition.face_encodings(image_test)[0]
cv.rectangle(image_test,(faceLocTest[0],faceLocTest[3]),(faceLocTest[1],faceLocTest[2]),(0,0,255),2)

result=face_recognition.compare_faces([encode_tusher],encode_test)
dist=face_recognition.face_distance([encode_tusher],encode_test)
print(result,dist)

cv.imshow("real",image_tusher)
cv.imshow("BRG",image_test)
cv.waitKey(0)

