#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import cv2

cap = cv2.VideoCapture(0)
image1 = cv2.imread('/Users/lakshmipriya/Desktop/pika.jpg')

while True:
    flag, frame = cap.read()
    if not flag:
        print("camera is not accessible")
        break
    
    
    image1=cv2.resize(image1,(frame.shape[1],frame.shape[0]))
    blended_image = cv2.addWeighted(frame, 0.8, image1, 0.2, gamma=0.2)
    cv2.imshow("blended image",blended_image)
    if cv2.waitKey(10) & 0xff == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

