import cv2
import sys
import json
from keras.layers.convolutional import Convolution2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.models import Sequential
from keras.optimizers import SGD
#from sklearn.cross_validation import train
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

import struct
import numpy as np
import cv2


from PIL.ImageWin import Window


def posReader(name): #reading look_vector
    with open(name) as f:
        data = json.load(f)
    return data["eye_details"]["look_vec"]

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

    model.add(Activation("softmax"))#LINEAR FOR COORDINATES

    model.compile(
        #loss="sparse_categorical_crossentropy",
        #optimizer=SGD(lr=0.01),
        loss='mse',#rmsprop
        optimizer='adam',
        metrics=["accuracy"])

    model.fit(
        train_data,
        train_labels,
        batch_size=128,
        nb_epoch=10)

    (loss, accuracy) = model.evaluate(
        test_data,
        test_labels,
        batch_size=128,
        verbose=1
    )
    model.save('my_model_50percent.h5')
    return loss,accuracy

def img_getter(add,size,weight, height):
    out = []
    for i in range(size):
        img = cv2.imread('eyes/'+str(i+1)+add) #READ EYE IMAGES
        if img is None: # IF NO EYE WAS DETECTED GET THE MAIN IMAGE
            img = cv2.imread('imgs/' + str(i + 1) + '.jpg')
            mean,std = cv2.meanStdDev(img, mask=None)#NORMALIZATION
            img[:,:,0] = (img[:,:,0] -mean[0])/std[0]
            img[:,:,1] = (img[:,:,1] -mean[1])/std[1]
            img[:,:,2] = (img[:,:,2] -mean[2])/std[2]
            reshapedimage = cv2.resize(img, (weight, height))# RESHAPING
            out.append(reshapedimage)
            continue
        mean, std = cv2.meanStdDev(img, mask=None) #NORMALIZATION
        img[:, :, 0] = (img[:, :, 0] - mean[0]) / std[0]
        img[:, :, 1] = (img[:, :, 1] - mean[1]) / std[1]
        img[:, :, 2] = (img[:, :, 2] - mean[2]) / std[2]
        reshapedimage = cv2.resize(img, (weight, height)) # RESHAPING
        out.append(reshapedimage)
    return out

def dataset_seperator(img,label,size, bool): # SEPERATE TO TRAIN AND TEST
    cord = []
    if bool:
        dist = []
        for i in range(size):
            d = np.linalg.norm(label[i] - label[:, None], axis=-1)  # FIND THE DISTANCE BETWEEN EACH IMAGE PAIR
            dist.append(np.sum(d))
        K = len(label) - int(len(label) * 0.005)  # INDEX OF N TOP IMAGES WITH MAX DISTANCE SUM
        threshold = dist
        threshold.sort()
        thresh = threshold[K]
        cord = []
        for i in range(size):  # GET LABEL OF N TOP IMAGES WITH MAX DISTANCE SUM
            if dist[i] > thresh:
                cord.append(label[i])
        print("Number of Classes:")
        print(len(cord))
        label2 = []
        for i in range(len(img)):  # SET THE CATAGORY FOR EACH IMAGE
            d = np.linalg.norm(label[i] - cord[:], axis=-1)
            index = np.argmin(d)
            label2.append(label[index])
        label = label2
    X_train, X_test, y_train, y_test = train_test_split(img, label, test_size=0.2)# SPLIT DATA TEST SIZE = 20%
    return X_train, X_test, y_train, y_test,cord


out = []
def label_reader(add,size,x):# READ LOOK VECTOR FROM JSON FILE AND SET AS LABEL
    pos = []
    for i in  range(size):
        pos.append(posReader(add+'/'+ str(i+1) + '.json').replace('(', '').replace(')', '').split(","))
    for i  in range(len(pos)):#SET X Y Z OF IMAGE
        out[i+x][0] = float(pos[i][0])
        out[i+x][1] = float(pos[i][1])
        out[i+x][2] = float(pos[i][2])

img1 = img_getter('-10.jpg',10267,100,100)
#img2 = img_getter('-20.jpg',10203,100,100)
#img3 = img_getter('-30.jpg',21610,100,100)
img =img1#+img2+img3
out = np.zeros([len(img),3])
l1 = label_reader('imgs-10',10267,0)
#l2 = label_reader('imgs-20',10203,10267)
#l3 = label_reader('imgs',21610,20470)
label = out
print("shape og images and labes:")
print(np.shape(img))
print(np.shape(label))
bool = 1#SET ZERO TO KEEP ORIGINAL LABELS AND SET TO 1 TO SET EACH COORDINATE TO NEAREST CENTER'S COORDINATE
train, test, train_l, test_l,cord = dataset_seperator(img,label,len(label),bool)
print(np.shape(test_l))
print(LeNet(np.array(train), np.array(train_l), np.array(test), np.array(test_l),
            (100, 100, 3), 3))
