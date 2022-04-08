import cv2, face_recognition,keyboard
import numpy as np

cap = cv2.VideoCapture(0)
while True:
    success, img = cap.read()
    cv2.waitKey(1)
    if keyboard.is_pressed("esc"):
        break
    cv2.imshow("webcam", img)
cv2.imshow("webcam", img)
cv2.waitKey(0)
