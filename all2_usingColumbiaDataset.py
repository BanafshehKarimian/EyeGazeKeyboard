import cv2
import numpy as np
import persian
from keras.models import load_model
from scipy import stats

cap = '' #initialize video capture
left_counter=0  #counter for left movement
right_counter=0	#counter for right movement
th_value=5   #changeable threshold value


import tkinter as tk

root = tk.Tk()
screen_width = int(root.winfo_screenwidth()/2)
screen_height = int(root.winfo_screenheight()/2)

def eye_fram():
    ret, frame = cap.read()
    alpha, beta, _ = np.shape(frame)
    frame = frame[alpha-250:alpha-150, int(beta / 2) - 250:int(beta / 2) + 250]
    framcolored = frame
    if ret == True:
        col = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        pupilFrame = frame
        eyes = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
        detected = eyes.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in detected:  # similar to face detection
            #pupilFrame = cv2.equalizeHist(frame[int(y + (h * .25)):int(y + h),
            #                              int(x):int(x + w)])  # using histogram equalization of better image.
            pupilFrame = framcolored[int(y + (h * .25))-10:int(y + h)+10,int(x)-10:int(x + w)+10]
        cv2.imshow('image', frame)
    return pupilFrame,frame



def pred_coord(pupil,model): #PREDICT THE AREA BETWEEN 21 CATEGORIES
    reshaped = cv2.resize(pupil, (100, 100))
    model_out = model.predict(np.array([reshaped, ]))[0]
    min_index = np.argmin(model_out)
    return min_index+1


def value(i,text):#CHECK FOR CHARACTER TO TYPE
    v = ''
    if i ==1:
        v='ه'
    elif i ==2:
        v = 'ی'
    elif i ==3:
        v = '.'
    elif i ==4:
        v = '.'
    elif i ==5:
        v = 'و'
    elif i ==6:
        v = 'ب'
    elif i ==7:
        v = 'ن'
    elif i ==8:
        v = 'ک'
    elif i ==9:
        v = 'گ'
    elif i ==10:
        v = 'ل'
    elif i ==11:
        v = 'م'
    elif i ==12:
        v = 'غ'
    elif i ==13:
        v = 'ع'
    elif i ==14:
        v = 'ظ'
    elif i ==15:
        v = '٪'
    elif i ==16:
        v = ' '
    elif i ==17:
        print('back')
        #if len(text)!=0:
        #    text = text[0:len(text)-2]
    elif i ==18:
        v = ''
    elif i ==19:
        v = ':'
    elif i ==20:
        v = 'آ'
    elif i ==21:
        v = '؟'
    #text = text + v
    return v

model = load_model('columbia_model_epoch_10_all.h5')
cap =cv2.VideoCapture(0)
keyboard = cv2.imread('keyB1.png')
seconds = 1
fps = 8#cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = fps * seconds

while 1:
    pupilFrame,frame = eye_fram()
    if np.shape(pupilFrame)!=np.shape(frame):
        pupil = pupilFrame
        break
frameId =0
text ='س'
label = np.zeros(multiplier)
while 1:
    frameId = frameId+1  # current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    success, image = cap.read()
    pupilFrame, frame = eye_fram()
    if np.shape(pupilFrame) != np.shape(frame):
        pupil = pupilFrame
    cv2.imshow('right eye', pupil)  # show last eye frame
    xy = pred_coord(pupil, model)
    label[frameId % multiplier] = xy

    if frameId%multiplier ==0:# IF EYE IS DETECTED
        mod = stats.mode(label)
        #print(mod.mode[0])
        text = text + value(xy,text)#int(mod.mode[0]), text)
        print(persian.convert_ar_characters(text))
        k = np.copy(keyboard)
        cv2.imshow('board', k)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
