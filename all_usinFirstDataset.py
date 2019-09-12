import cv2
import numpy as np
from keras.models import load_model

cap = '' #initialize video capture
left_counter=0  #counter for left movement
right_counter=0	#counter for right movement
th_value=5   #changeable threshold value




def eye_fram():# DETECT THE EYE
    ret, frame = cap.read()
    alpha, beta, _ = np.shape(frame)
    frame = frame[100:200, int(beta / 2) - 250:int(beta / 2) + 250]
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



def get_first(): # SAVE IMAGE BY PRESSING SPACE AND GETTING THE 4 CORNER
    cam = cv2.VideoCapture(0)
    cv2.namedWindow("Calibration")
    img_counter = 0
    image = cv2.imread('keyB1.png')
    cv2.imshow('board', image)
    res=[]
    while True:
        ret, frame = cam.read()
        alpha, beta, _ = np.shape(frame)
        frame = frame[100:200, int(beta / 2) - 250:int(beta / 2) + 250]
        framcolored = frame
        if ret == True:
            col = frame
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            pupilFrame = frame
            eyes = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
            detected = eyes.detectMultiScale(frame, 1.3, 5)
            for (x, y, w, h) in detected:  # similar to face detection
                # pupilFrame = cv2.equalizeHist(frame[int(y + (h * .25)):int(y + h),
                #                              int(x):int(x + w)])  # using histogram equalization of better image.
                pupilFrame = framcolored[int(y + (h * .25)) - 10:int(y + h) + 10, int(x) - 10:int(x + w) + 10]

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
            cv2.imwrite(img_name, pupilFrame)
            res.append(pupilFrame)
            print("{} written!".format(img_name))
            img_counter += 1
            if img_counter > 3:
                break

    cam.release()
    cv2.destroyAllWindows()
    return res

def calibrator(im0_0,im0_1,im1_0,im1_1,model):#CALIBRATION: CALCULATE M
    im0_0 = cv2.resize(im0_0, (100, 100))
    im0_1 = cv2.resize(im0_1, (100, 100))
    im1_0 = cv2.resize(im1_0, (100, 100))
    im1_1 = cv2.resize(im1_1, (100, 100))
    c0_0 = model.predict(np.array([im0_0, ]))[0]
    c0_1 = model.predict(np.array([im0_1, ]))[0]
    c1_0 = model.predict(np.array([im1_0, ]))[0]
    c1_1 = model.predict(np.array([im1_1, ]))[0]  # c0_0,c0_1,c1_0,c1_1
    pts1 = np.float32([[c0_0[1], c0_0[2]], [c0_1[1], c0_1[2]], [c1_0[1], c1_0[2]], [c1_1[1], c1_1[2]]])
    pts2 = np.float32([[0, 0], [0, 1500], [670, 0], [670, 1500]])
    M = cv2.getPerspectiveTransform(pts1, pts2)
    return M
def pred_coord(pupil,model,M): #PREDICT AND TRANSFER USING M
    reshaped = cv2.resize(pupil, (100, 100))
    model_out = model.predict(np.array([reshaped, ]))[0]
    pts = np.array([[model_out[1], model_out[2]]], dtype="float32")
    pts = np.array([pts])
    xy = cv2.warpPerspective(pts, M, (1, 1))[0][0]  # predict point of look
    return xy


model = load_model('my_model_epoch_10.h5') #LOAD MODEL
get_first()#GET 4 CORNER'S COORDINATE

#LOAD THEM
im0_0=cv2.imread('0.png')
im1_0=cv2.imread('1.png')
im0_1=cv2.imread('2.png')
im1_1=cv2.imread('3.png')
M = calibrator(im0_0,im0_1,im1_0,im1_1,model)#CALIBRATE USING PROJECTIVE TRANSFORM
cap =cv2.VideoCapture(0)
image = cv2.imread('keyB1.png')

while 1:#WAIT FOR EYE TO DETECT
    pupilFrame,frame = eye_fram()
    if np.shape(pupilFrame)!=np.shape(frame):
        pupil = pupilFrame
        break
while 1:
    pupilFrame, frame = eye_fram()
    if np.shape(pupilFrame) != np.shape(frame):# CHECK IF THER IS AN EYE
        pupil = pupilFrame
        cv2.imshow('right eye', pupil)#show last eye frame
        xy= pred_coord(pupil,model,M)#PREDICT COORDINATE
        print(xy)
        cv2.circle(image, (int(xy[0]*10), int(xy[1]*10)), 10, (0, 0, 255), 2) #CIRCLE THE KEYBOARD
        cv2.imshow('board', image)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
cap.release()
cv2.destroyAllWindows()
