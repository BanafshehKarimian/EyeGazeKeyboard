from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from sklearn.model_selection import train_test_split
import cv2
import numpy as np
import glob
from os import listdir
from os.path import isfile, join
import numpy
import cv2


def eye_fram(frame): # EXTRACTIN THE EYE PART USING HAR CASCADE
    framcolored = frame
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
    pupilFrame = frame
    eyes = cv2.CascadeClassifier('haarcascade_righteye_2splits.xml')
    detected = eyes.detectMultiScale(frame, 1.3, 5)
    for (x, y, w, h) in detected:  # similar to face detection
            #pupilFrame = cv2.equalizeHist(frame[int(y + (h * .25)):int(y + h),
            #                              int(x):int(x + w)])  # using histogram equalization of better image.
        pupilFrame = framcolored[int(y + (h * .25))-10:int(y + h)+10,int(x)-10:int(x + w)+10]
    #    cv2.imshow('image', frame)
    return pupilFrame#,frame

def address(mypath):
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    images = numpy.empty(len(onlyfiles), dtype=object)
    for n in range(0, len(onlyfiles)):
        images[n] = cv2.imread(join(mypath, onlyfiles[n]))
    return images

def img_getter(weight, height): # CODE FOR READING FROM FILE AND EXTRACTING THE EYE
    out = []
    label = []
    counter = 0
    ww = 1
    v = 0
    h = 0
    for i in range(8):
        add = "columia/"+str(ww)+"/000"+str(i + 1)+"_2m_0P_"+str(v)+"V_"+str(h)+"H.jpg"
        add1 = "columia/"+str(ww)+"/000"+str(i + 1)+"_2m_15P_"+str(v)+"V_"+str(h)+"H.jpg"
        add2 = "columia/"+str(ww)+"/000"+str(i + 1)+"_2m_-15P_"+str(v)+"V_"+str(h)+"H.jpg"
        img = cv2.imread(add)
        print(add)
        eye = eye_fram(img)
        if eye is not  None:
            reshapedimage = cv2.resize(eye, (weight, height))
            out.append(reshapedimage)
            label.append(i)
            cv2.imwrite(str(counter+1)+'.png',reshapedimage)
            counter = counter+1
        img = cv2.imread(add1)
        # print(images)
        # for img in images:
        print(add1)
        eye = eye_fram(img)
        if eye is None:
            print("none")
        else:
            reshapedimage = cv2.resize(eye, (weight, height))
            out.append(reshapedimage)
            label.append(i)
            cv2.imwrite(str(counter+1)+'.png',reshapedimage)
            counter = counter+1
        img = cv2.imread(add2)
        # print(images)
        # for img in images:
        eye = eye_fram(img)
        print(add2)
        if eye is not None:
            reshapedimage = cv2.resize(eye, (weight, height))
            out.append(reshapedimage)
            label.append(i)
            cv2.imwrite(str(counter+1)+'.png',reshapedimage)
            counter = counter+1
    for i in range(31):
        add = "columia/"+str(ww)+"/00" + str(i + 10) + "_2m_0P_"+str(v)+"V_"+str(h)+"H.jpg"
        add1 = "columia/"+str(ww)+"/00" + str(i + 10) + "_2m_15P_"+str(v)+"V_"+str(h)+"H.jpg"
        add2 = "columia/"+str(ww)+"/00" + str(i + 10) + "_2m_-15P_"+str(v)+"V_"+str(h)+"H.jpg"
        img = cv2.imread(add)
        # print(images)
        # for img in images:
        print(add)
        eye = eye_fram(img)
        if eye is not None:
            reshapedimage = cv2.resize(eye, (weight, height))
            out.append(reshapedimage)
            label.append(i)
            cv2.imwrite(str(counter + 1) + '.png', reshapedimage)
            counter = counter + 1
        img = cv2.imread(add1)
        # print(images)
        # for img in images:
        print(add1)
        eye = eye_fram(img)
        if eye is None:
            print("none")
        else:
            reshapedimage = cv2.resize(eye, (weight, height))
            out.append(reshapedimage)
            label.append(i)
            cv2.imwrite(str(counter + 1) + '.png', reshapedimage)
            counter = counter + 1
        img = cv2.imread(add2)
        # print(images)
        # for img in images:
        eye = eye_fram(img)
        print(add2)
        if eye is not None:
            reshapedimage = cv2.resize(eye, (weight, height))
            out.append(reshapedimage)
            label.append(i)
            cv2.imwrite(str(counter + 1) + '.png', reshapedimage)
            counter = counter + 1
    return out,label


def LeNet(train_data, train_labels,test_data, test_labels,inps,d):
    model = Sequential()
    #1
    model.add(Convolution2D(
        filters=32,
        kernel_size=(7, 7),
        padding="same",
        strides=(1, 1),
        input_shape=inps))

    model.add(Activation(
        activation="relu"))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)))
    #2
    model.add(Convolution2D(
        filters=32,
        kernel_size=(5, 5),
        strides=(1, 1),
        padding="same"))

    model.add(Activation(
        activation="relu"))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)))
    # 2

    model.add(Convolution2D(
        filters=64,
        kernel_size=(3, 3),
        strides=(1, 1),
        padding="same"))

    model.add(Activation(
        activation="relu"))

    model.add(MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2)))

    model.add(Flatten())
    #3
    #model.add(Dense(128))

    #model.add(Activation(
    #    activation="relu"))
    #4
    #model.add(Dense(128))

    #model.add(Activation(
    #    activation="relu"))
    #5
    model.add(Dense(100))
    model.add(Dense(d))

    model.add(Activation("softmax"))

    model.compile(
        loss="categorical_crossentropy",
        optimizer=SGD(lr=0.0001))
        #loss='mse',#rmsprop
        #optimizer='rmsprop',
        #metrics=["accuracy"])

    model.fit(
        train_data,
        train_labels,
        batch_size=128,
        nb_epoch=100,
        validation_split=0.1)

    (loss, accuracy) = model.evaluate(
        test_data,
        test_labels,
        batch_size=128,
        verbose=1
    )
    model.save('columbia_model_epoch_100_2_all.h5')
    return loss,accuracy
from keras.utils import to_categorical


def get_img_label():# GET LABELS (21 POSITION CLASSES)
    label = []
    out = []
    for i in range(21):
        for j in range(117):
            print(str(i+1)+"/"+str(j+1)+".png")
            img = cv2.imread(str(i+1)+"/"+str(j+1)+".png")
            if img is None:
                print("None")
                continue
            mean, std = cv2.meanStdDev(img, mask=None)
            img[:, :, 0] = (img[:, :, 0] - mean[0]) / std[0]
            img[:, :, 1] = (img[:, :, 1] - mean[1]) / std[1]
            img[:, :, 2] = (img[:, :, 2] - mean[2]) / std[2]
            label.append(i)
            kernel = np.ones((10, 10), np.uint8)
            closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
            reshapedimage = cv2.resize(closing, (100, 100))
            out.append(reshapedimage)
    X_train, X_test, y_train, y_test = train_test_split(out, label, test_size=0.2)
    return X_train, X_test, to_categorical(y_train), to_categorical(y_test)
def otsu(gray): # OTSU CODE FOR USING
    pixel_number = gray.shape[0] * gray.shape[1]
    mean_weigth = 1.0/pixel_number
    his, bins = np.histogram(gray, np.array(range(0, 256)))
    final_thresh = -1
    final_value = -1
    for t in range(255):
        value = np.sum(his[t:])*np.sum(his[:t])* (mean_weigth**2)*((np.mean(his[:t]) - np.mean(his[t:])) ** 2)
        if value > final_value:
            final_thresh = t
            final_value = value
    final_img = gray.copy()
    for i in range(len(gray)):
        for j in range(len(gray[0])):
            if gray[i][j] > final_thresh:
                final_img[i][j] = 0
            else:
                final_img[i][j] = 255
    return final_img,final_thresh

def get_img_label_black_wight():# GET BLACK AND WIGHT IMAGE + OTSU
    label = []
    out = []
    for i in range(21):
        for j in range(117):
            print(str(i+1)+"/"+str(j+1)+".png")
            img = cv2.imread(str(i+1)+"/"+str(j+1)+".png",0)
            if img is None:
                print("None")
                continue
            mean, std = cv2.meanStdDev(img, mask=None)
            img[:, :] = (img[:, :] - mean) / std
            img , _ = otsu(img)
            label.append(i)
            reshapedimage = cv2.resize(img, (100, 100))
            reshapedimage = reshapedimage[..., np.newaxis]
            print(np.shape(reshapedimage))
            out.append(reshapedimage)
    X_train, X_test, y_train, y_test = train_test_split(out, label, test_size=0.2)
    np.array(X_train).reshape(-1, 100, 100, 1)
    np.array(X_test).reshape(-1, 100, 100, 1)
    print(np.shape(X_train))
    #np.reshape(X_train,(np.shape(X_train)[0],np.shape(X_train)[1],1))
    return X_train, X_test, to_categorical(y_train), to_categorical(y_test)


#x , y = img_getter(100,100)
black = 0
d = (100, 100,3)
if black:
    d = (100,100,1)
    train, test, train_l, test_l = get_img_label_black_wight()
else:
    train, test, train_l, test_l = get_img_label()
print(np.shape(test_l))
print(LeNet(np.array(train), np.array(train_l), np.array(test), np.array(test_l), d, 21))
