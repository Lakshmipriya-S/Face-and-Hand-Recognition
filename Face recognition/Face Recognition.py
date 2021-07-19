#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#Import libraries
import cv2
import face_recognition
import numpy as np
from face_recognition.api import face_distance, face_encodings, face_locations


# Reading an image file from teh local system and trying to detect the face on that picture

lachu = face_recognition.load_image_file("/Users/lakshmipriya/Desktop/lachu.jpg")
encodings = face_recognition.face_encodings(lachu)[0]


#Saving the encodings of the face from the photos in a variable, so that it can recognise that person in live photos
known_face_encodings = [ encodings ]


#Labeling that person as "Lachu", if the known encodings match teh live camera image
known_face_names = ["lachu"]


#Capturing the video through default first camera - 0
cap = cv2.VideoCapture(0)


# Running a while loop to capture teh sttaus of the camera and the frame size of the image taken form the camera
# Also the loops contains certain conditions to be executed if, the condition is true, such as, resizing and capturing the locations of teh face with landmarks
while cap.isOpened():
    success,frame =cap.read()
    if not success:
        print("could not access the camera")
        break
        
        
    small_frame = cv2.resize(frame,(0,0), fx = 1/4, fy = 1/4)
    face_locations = face_recognition.face_locations(small_frame)
    face_encodings = face_recognition.face_encodings(small_frame,face_locations)
    face_names = []

# Checking if the face encodings form camera mactches the encodings from the trained image encodings from the local file system
    for face_encoding in face_encodings:

        matches = face_recognition.compare_faces(known_face_encodings,face_encoding)
        name = "unkown"
        face_distances = face_recognition.face_distance(known_face_encodings,face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

            face_names.append(name)

            print(face_names)

# When teh condition matches, draw a rectangle box on teh recognized face and label the person
    for (top, right, bottom, left), name in zip(face_locations, face_names):

        top *= 4
        right *= 4
        bottom *= 4
        left *= 4
    
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)


        cv2.imshow("frame",frame)

# Checking condition to break the loop and capture image frame for every 10 micro second
        if cv2.waitKey(10) & 0xff == ord("q"):
            break


# release the captured image and destroyed all the opened windows
cap.release()
cv2.destroyAllWindows()


# In[ ]:


#pip install face_recognition

