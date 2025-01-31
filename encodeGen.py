import cv2
import face_recognition
import pickle
import os

# Load the student images
folderImagePath = 'images'
modeImageList = os.listdir(folderImagePath)

# Storing images and id in a list
imgList = []
studentIds = []

for path in modeImageList:
    imgList.append(cv2.imread(folderImagePath+'/'+path))
    studentIds.append(os.path.splitext(path)[0])


def findEncodings(imageList):
    encodeList = []
    for img in imageList:
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encode  = face_recognition.face_encodings(img)[0]
        encodeList.append(encode)
        
    return encodeList


print('Encoding Started')
encodeListKnown = findEncodings(imgList)
encodeListKnownWithIds = [encodeListKnown, studentIds]
print('Encoding Complete')

file = open('EncodeFile.p', 'wb')
pickle.dump(encodeListKnownWithIds,file)
file.close()

print('File Saved')
