import numpy as np
import cv2
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import load_model
from time import sleep
import os
import pyautogui
import time
#import serial
## from gpiozero import LED
#ser = serial.Serial("COM5")
#print(ser.name)
# bioled=LED(3, False)
# nonbioled=LED(2, False)

timedef = 0.01
#Flag = True


labels=['1', 'L', 'OK', 'Palm', 'Peace']
model = load_model('final_model.h5')
font                   = cv2.FONT_HERSHEY_SIMPLEX
bottomLeftCornerOfText = (10,50)
fontScale              = 1
fontColor              = (255,255,255)
lineType               = 2
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

print(pyautogui.size())
#import pyautogui
#pyautogui.moveTo(100, 100, duration = 1)


def start():
    text='nothing yet'
    while(True):

        # Capture frame-by-frame
        ret, frame = cap.read()

        # Our operations on the frame come here
        frame=cv2.flip(frame, flipCode=1)
        cv2.imwrite('./pic.jpg', frame)
        img = image.load_img('./pic.jpg', target_size=(224, 224))
        x=image.img_to_array(img)
        x=np.expand_dims(x, axis=0)
        images=np.vstack([x])
        results = model.predict(images)
        idx=np.where(results[0]==np.amax(results[0]))
        print(type(results[0]))
        print(results[0])

        cv2.putText(frame, text, bottomLeftCornerOfText, font, fontScale, (0,0,0), lineType+2)
        cv2.putText(frame, text, bottomLeftCornerOfText, font, fontScale, (255,255,255), lineType)

        if (labels[idx[0][0]]=='Peace'):
            text='Gesture: Peace'
            print(text)
            #ser.write(b'A')
            time.sleep(timedef)
        elif (labels[idx[0][0]]=='L'):
            text='Gesture: L'
            print(text)
            time.sleep(timedef)
        elif (labels[idx[0][0]]=='OK'):
            text='Gesture: OK'
            print(text)
            #ser.write(b'J')
            time.sleep(1)
            pyautogui.hotkey("alt", "a")
            time.sleep(1)
            #flag = False
            break
        elif (labels[idx[0][0]]=='Palm'):
            text='Gesture: Palm'
            print(text)
            time.sleep(timedef)
        else:
            text='No gesture'
            print(text)
        # Display the resulting frame
        cv2.imshow('frame', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

start()
cap.release()
cv2.destroyAllWindows()