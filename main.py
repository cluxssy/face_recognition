import cv2
import requests
import numpy as np
import os
import pickle
import face_recognition
import cvzone

# Define the background image path
background_path = 'Resources/background.png'

# Load the background image
imgBackground = cv2.imread(background_path)

# Load the mode images
folderModePath = 'Resources/Modes'
modePathList = os.listdir(folderModePath)
imgModeList = []
for path in modePathList:
    imgModeList.append(cv2.imread(folderModePath+'/'+path))

# Load the encoding file
print('Loading Encode File ...')
file = open('EncodeFile.p', 'rb')
encodeListKnownWithIds = pickle.load(file)
encodeListKnown, studentIds = encodeListKnownWithIds
print('Encode File Loaded')

# Define the IP Camera URL (Replace with your actual URL)
url = "http://192.168.0.102:8080/shot.jpg"

print("🔄 Fetching camera feed... Press 'q' to exit.")

while True:
    try:
        # Fetch the image from the IP camera
        img_resp = requests.get(url, timeout=5)

        # Convert response content to an image
        img_arr = np.array(bytearray(img_resp.content), dtype=np.uint8)
        img = cv2.imdecode(img_arr, -1)

        # Making images ready for face recognition
        imgS = cv2.resize(img,(0,0),None,0.25,0.25)
        imgS = cv2.cvtColor(imgS,cv2.COLOR_BGR2RGB)

        # Find faces in the current frame and their encodings
        faceCurFrame = face_recognition.face_locations(imgS)
        encodeCurFrame = face_recognition.face_encodings(imgS,faceCurFrame)

        # Resize the image for display
        img = cv2.resize(img, (640, 480))
        

        # Overlay the background image
        imgBackground[162:162+480, 55:55+640] = img
        imgBackground[44:44+633, 808:808+414] = imgModeList[0]

        for encodeFace, faceloc in zip(encodeCurFrame, faceCurFrame):
            matches = face_recognition.compare_faces(encodeListKnown,encodeFace)
            faceDis = face_recognition.face_distance(encodeListKnown,encodeFace)

            matchIndex = np.argmin(faceDis)
            # print("MATCH INDEX",matchIndex)

            if matches[matchIndex]:
                print("Face Detected"+str(studentIds[matchIndex]))
                y1,x2,y2,x1 = faceloc
                y1,x2,y2,x1 = y1*4,x2*4,y2*4,x1*4
                bbox = 55+x1,162+y1,x2-x1,y2-y1
                imgBackground=cvzone.cornerRect(imgBackground, bbox,rt=0)

        # Show the webcam feed and background image
        cv2.imshow("Webcam Feed", img)
        cv2.imshow("Face Attendance", imgBackground)

        

    except Exception as e:
        print(f"❌ Error: {e}")

    # Exit the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        print("🛑 Exiting program...")
        break

