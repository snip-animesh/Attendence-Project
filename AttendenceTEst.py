import cv2 as cv, face_recognition, os, keyboard
import numpy as np
from datetime import datetime


def rescaleFrame(frame, scale=0.15):
    # Image ,Videos and  Live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


def getAttendence(name,roll):
    with open('Attendence1.csv', 'r+') as f:
        myDataList = f.readlines()
        nameList = []
        # print(myDataList)
        for line in myDataList:
            entry = line.split(',')
            nameList.append(entry[0])
        if name not in nameList:
            now = datetime.now()
            dtString = now.strftime('%H:%M:%S')
            f.writelines(f'\n{name},{roll},{dtString}')


DIR = r'D:\python Code\Attendence Project\peoples'
myList = os.listdir(DIR)
images = []
classNames = []
# print(myList)
for cls in myList:
    curIm = cv.imread(f'{DIR}/{cls}')
    curIm = rescaleFrame(curIm)
    images.append(curIm)
    # cv.imshow("im",curIm)
    # cv.waitKey(0)
    classNames.append(os.path.splitext(cls)[0])


def findEncodings(images):
    encodelist = []
    for img in images:
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist.append(encode)
    return encodelist


# Find encode list for known faces
encodeKnown = findEncodings(images)
print('Encoding completed !!')

# def getImage():
cap = cv.VideoCapture(1)
while True:
    success, img = cap.read()
    cv.waitKey(1)
    # imgS=cv.resize(img,(0,0),None,0.25,0.25)
    imgS = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    if keyboard.is_pressed("esc"):
        break
    cv.imshow("webcam", img)
    cv.waitKey(1)
    # return imgS, img

    # get image from webcam
    # imgS, img = getImage()
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
        if matches[matchIndex]:
            nameRoll = classNames[matchIndex].upper()
            name,roll=nameRoll.split('_')
            # print(name)
            y1, x2, y2, x1 = faceLoc
            cv.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv.rectangle(img, (x1, y2 - 35), (x2, y2), (0, 255, 0), cv.FILLED)
            cv.putText(img, name, (x1 + 6, y2 - 6), cv.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
            cv.imshow("Detected Face", img)
            getAttendence(name,roll)

cv.waitKey(0)

# This is what I learnt