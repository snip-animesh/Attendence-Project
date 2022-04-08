import cv2 as cv, face_recognition, os
import pickle


def rescaleFrame(frame, scale=0.15):
    # Image ,Videos and  Live videos
    width = int(frame.shape[1] * scale)
    height = int(frame.shape[0] * scale)
    dimension = (width, height)
    return cv.resize(frame, dimension, interpolation=cv.INTER_AREA)


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
print(classNames)

def findEncodings(images):
    encodelist = {}
    for img, name in zip(images, classNames):
        img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        encode = face_recognition.face_encodings(img)[0]
        encodelist[name] = encode
    return encodelist


# Find encode list for known faces
encodeKnown = findEncodings(images)
print('Encoding completed !!')

with open('dataset_faces.dat', 'wb') as f:
    pickle.dump(encodeKnown, f)
