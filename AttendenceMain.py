import cv2 as cv, face_recognition, keyboard
import numpy as np, pickle
from datetime import datetime

with open('dataset_faces.dat', 'rb') as f:
    encodeKnown = pickle.load(f)
# print(encodeKnown)
classNames = list(encodeKnown.keys())
encodeKnown = np.array(list(encodeKnown.values()))


def getImage():
    cap = cv.VideoCapture(1)
    while True:
        success, img = cap.read()
        cv.waitKey(1)
        # imgS=cv.resize(img,(0,0),None,0.25,0.25)
        imgS = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        if keyboard.is_pressed("esc"):
            break
        cv.imshow("webcam", img)
    return imgS, img


def getAttendence(name,roll):
    with open('Attendence.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{roll},{dtString}')


# get image from webcam
imgS, img = getImage()
# detect face
facesCurFrame = face_recognition.face_locations(imgS)
encodesCurFrame = face_recognition.face_encodings(imgS, facesCurFrame)
for encodeFace, faceLoc in zip(encodesCurFrame, facesCurFrame):
    # find matching
    matches = face_recognition.compare_faces(encodeKnown, encodeFace)
    # find distance
    faceDis = face_recognition.face_distance(encodeKnown, encodeFace)
    # print(faceDis)
    # find best match (lowest distance)
    matchIndex = np.argmin(faceDis)
    print(matchIndex)
    if matches[matchIndex]:
        nameRoll = classNames[matchIndex].upper()
        name,roll=nameRoll.split('_')
        print(name,roll)
        y1, x2, y2, x1 = faceLoc
        cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv.rectangle(img, (x1, y2 - 20), (x2, y2), (0, 255, 0), cv.FILLED)
        cv.putText(img, name, (x1 + 6, y2), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
        cv.imshow("Detected Face", img)
        getAttendence(name,roll)
        cv.waitKey(1)

# cv.destroyAllWindows()
cv.waitKey(0)
# My application