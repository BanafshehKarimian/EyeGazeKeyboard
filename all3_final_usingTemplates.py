from statistics import stdev

import cv2
import numpy as np
from keras.models import load_model
from scipy import stats
import arabic_reshaper
import persian
cap = '' #initialize video capture
left_counter=0  #counter for left movement
right_counter=0	#counter for right movement
th_value=5   #changeable threshold value


import tkinter as tk

root = tk.Tk()
screen_width = int(root.winfo_screenwidth()/2)
screen_height = int(root.winfo_screenheight()/2)

def eye_fram():# DETECTING THE EYE
    ret, frame = cap.read()
    alpha, beta, _ = np.shape(frame)
    frame = frame[:, int(beta / 2) - 250:int(beta / 2) + 250]
    framcolored = frame
    if ret == True:
        col = frame
        frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        pupilFrame = frame
        eyes = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
        detected = eyes.detectMultiScale(frame, 1.3, 5)
        for (x, y, w, h) in detected:  # similar to face detection
            pupilFrame = cv2.equalizeHist(frame[int(y + (h * .25)):int(y + h),
                                          int(x):int(x + w)])  # using histogram equalization of better image.
            #pupilFrame = framcolored[int(y + (h * .25))-10:int(y + h)+10,int(x)-10:int(x + w)+10]
        cv2.imshow('image', frame)
    return pupilFrame,frame



def get_first():# GET THE 21 FIRST TEMPLATES
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Calibration")
    img_counter = 0
    image = cv2.imread('keyB1.png')
    cv2.imshow('board', image)
    res=[]
    while True:
        ret, frame = cam.read()
        alpha, beta, _ = np.shape(frame)
        frame = frame[100:200, 250:750]
        framcolored = frame
        if ret == True:
            col = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            pupilFrame = frame
            eyes = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
            detected = eyes.detectMultiScale(frame, 1.3, 5)
            for (x, y, w, h) in detected:  # similar to face detection
                 pupilFrame = cv2.equalizeHist(frame[int(y + (h * .25)):int(y + h),
                                              int(x):int(x + w)])  # using histogram equalization of better image.
                #pupilFrame = framcolored[int(y + (h * .25)) - 10:int(y + h) + 10, int(x) - 10:int(x + w) + 10]

        cv2.imshow("test", pupilFrame)
        if not ret:
            break
        k = cv2.waitKey(1)

        if k % 256 == 27:
            # ESC pressed
            print("Escape hit, closing...")
            break
        elif k % 256 == 32:
            # SPACE pressed
            img_name = str(img_counter)+".png"
            reshapedimage = cv2.resize(pupilFrame, (100, 100))
            cv2.imwrite(img_name, reshapedimage)
            res.append(pupilFrame)
            print("{} written!".format(img_name))
            img_counter += 1
            if img_counter > 20:
                break

    cam.release()
    cv2.destroyAllWindows()
    return res

def features(img):
    out = []
    _, cnt, _ = cv2.findContours(img, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    x, y, w, h = cv2.boundingRect(cnt[0])
    aspect_ratio = float(w) / h
    area = cv2.contourArea(cnt[0])
    equi_diameter = np.sqrt(4 * area / np.pi)
    rect_area = w * h
    extent = float(area) / rect_area
    #(_,_), (MA, ma), angle = cv2.fitEllipse(cnt[0])
    cnt = cnt[0]
    leftmostx,leftmosty = tuple(cnt[cnt[:, :, 0].argmin()][0])
    rightmostx,rightmosty = tuple(cnt[cnt[:, :, 0].argmax()][0])
    topmostx,topmosty = tuple(cnt[cnt[:, :, 1].argmin()][0])
    bottommostx,bottommosty = tuple(cnt[cnt[:, :, 1].argmax()][0])
    out.append(aspect_ratio)
    out.append(equi_diameter)
    out.append(extent)
    out.append(leftmostx)
    out.append(leftmosty)
    out.append(rightmostx)
    out.append(rightmosty)
    out.append(topmostx)
    out.append(topmosty)
    out.append(bottommostx)
    out.append(bottommosty)
    return  np.array(out)

def dif(imgs,img):# FIND DIFFERENCE USING L1
    min = np.sum(imgs[0]-img)
    index = 0
    for i in range(len(imgs)):
        if np.sum(imgs[i]-img)< min:
            index = i
            min = np.sum(imgs[i]-img)
    return index+1


def dif_featur(imgs_f,img_f):# FIND DIFFERENCE USING L1
    min = np.sum((imgs_f[0]-img_f)**2)
    index = 0
    for i in range(len(imgs)):
        if np.sum((imgs_f[0]-img_f)**2)< min:
            index = i
            min = np.sum((imgs_f[0]-img_f)**2)
    return index+1

def value(i,text):# RETURN THE VALUE FOR KEYBOARD
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
#get_first()
imgs = []
featur_list = []
for i in range(21):
    x = cv2.imread(str(i)+'.png',0)
    imgs.append(x)
    featur_list.append(features(x))
cap =cv2.VideoCapture(0)
keyboard = cv2.imread('keyB1.png')
seconds = 1
fps = 8#cap.get(cv2.CAP_PROP_FPS) # Gets the frames per second
multiplier = fps * seconds
text = 'س'
while 1:
    pupilFrame,frame = eye_fram()
    if np.shape(pupilFrame)!=np.shape(frame):
        pupil = pupilFrame
        break
frameId =0
label =np.zeros(multiplier)
while 1:
    frameId = frameId+1  # current frame number, rounded b/c sometimes you get frame intervals which aren't integers...this adds a little imprecision but is likely good enough
    success, image = cap.read()
    pupilFrame, frame = eye_fram()
    if np.shape(pupilFrame) != np.shape(frame):# PREDICT FOR 8 FRAMES AND SET THE ONE WITH THE MOST VOTES
        pupil = pupilFrame
        cv2.imshow('right eye', pupil)  # show last eye frame

        reshapedimage = cv2.resize(pupil, (100, 100))
        #xy = dif_featur(featur_list, features(reshapedimage))
        xy = dif(imgs, reshapedimage)
        label[frameId % multiplier] = xy
        if frameId % multiplier == 0:
            mod = stats.mode(label)
            text = text+value(int(mod.mode[0]),text)
            print(persian.convert_ar_characters(text))
            k = np.copy(keyboard)
            cv2.imshow('board', k)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
