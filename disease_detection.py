from keras.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.preprocessing.image import img_to_array
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import numpy as np
import argparse
import random
import cv2
import os
from skimage import io
import tensorflow as tf
import keras
from keras.models import Sequential
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dense
from keras.layers.core import Dropout
from keras import backend as K
from keras.utils.vis_utils import plot_model
from skimage import io
from skimage.transform import resize, rescale, rotate, setup, warp, AffineTransform
import pandas as pd
from sklearn.metrics import confusion_matrix
from keras.layers.normalization import BatchNormalization
from keras.models import load_model
import matplotlib.image as mpimg
from keras import optimizers
import random 



class net:
    @staticmethod
    def build(height,width,depth,num_classes):#Number of channels-1(grayscale),3(RGB)
        model=Sequential()
        shape=(height,width,depth)
        chanDim=-1
        #Channel last ordering is default for tensorflow
        if K.image_data_format()=="channels_first":
            shape=(depth,height,width)
            chanDim=1
        #To add layers for model->Layer1
        model.add(Conv2D(64,(3,3),padding="same",input_shape=shape))
        model.add(Conv2D(64,(3,3),padding="same"))
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        #model.add(Activation("relu"))#TO make sure there are no negative pixel values in the feature map
        #model.add(BatchNormalization(axis=chanDim))
        
        
        #Layer2
        model.add(Conv2D(128,(3,3),padding="same"))#The same padding means zero padding is provided,whereas in VALID->No zero padding,values are dropped 
        model.add(Conv2D(128,(3,3),padding="same"))
        #model.add(Activation("relu"))
        #model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3,3),strides=(2,2)))
        #layer 3
        model.add(Conv2D(256,(3,3),padding="same"))#The same padding means zero padding is provided,whereas in VALID->No zero padding,values are dropped 
        model.add(Conv2D(256,(3,3),padding="same"))
        model.add(Conv2D(256,(3,3),padding="same"))
        #model.add(Activation("relu"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #model.add(BatchNormalization(axis=chanDim))
        #model.add(Dropout(0.35))
        
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        #model.add(Activation("relu"))
        #model.add(BatchNormalization(axis=chanDim))

        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(Conv2D(512, (3, 3), padding="same"))
        model.add(MaxPooling2D(pool_size=(2,2),strides=(2,2)))
        
        #model.add(Conv2D(32, (3, 3), padding="same"))
        #model.add(Activation("relu"))
        #model.add(BatchNormalization(axis=chanDim))
        #model.add(MaxPooling2D(pool_size=(2, 2)))
        #model.add(Dropout(0.35))
        
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Dense(4096))
        #model.add(Activation("relu"))
        #model.add(BatchNormalization())
        #model.add(Dropout(0.35))
        model.add(Dense(num_classes))
        #model.add(Activation("softmax"))
        return model

total_epoch=12
batch_size=48
learning_rate=0.0001
seed=7


data=[]
labels=[]
temp=[]

for root,sub,files in os.walk('H:/dataset'):
    for name in files:
        num=os.path.join(root,name)
        data.append(num)
#This is done to shuffle images
#random.shuffle(data)   
for image in data:
    #im1=cv2.imread(image)
    #im1=cv2.cvtColor(im1,cv2.COLOR_BGR2GRAY)
    im1=io.imread(image)
    im2=resize(im1,(224,224,3))
    im3=img_to_array(im2)
    temp.append(im3)
#temp contains images
for path in data:
    path=path.split(os.path.sep)[-2]
    '''if path=='not_tomato':
        label=0
        labels.append(label)'''
    if path=='tomato_curl':
        label=0
        labels.append(label)
    elif path=='tomato_dspot':
        label=1
        labels.append(label)
    elif path=='tomato_healthy':
        label=2
        labels.append(label)


#labels contains labels

#After obtaining images and labels,split them into train and test
#scale the intensities to [0,1]
temp_array=np.array(temp,dtype="float")/255.0
label_array=np.array(labels)
(trainX,testX,trainY,testY)=train_test_split(temp_array,label_array,test_size=0.30,random_state=42)
#Convert integers to vectors
trainY= pd.get_dummies(trainY).values
testY=pd.get_dummies(testY).values#get them in the form of one hot labels in array
#matrix multiplication with np.zeros

#Next is data augmentation-TO increase the number of samples
aug = ImageDataGenerator(rotation_range=30, width_shift_range=0.1,
	height_shift_range=0.1, shear_range=0.2, zoom_range=0.2,
	horizontal_flip=True, fill_mode="nearest")
#kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)

print('Compiling model....')
model=net.build(height=224,width=224,depth=3,num_classes=3)
opt=Adam(lr=learning_rate,decay=learning_rate/total_epoch)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])#Sometimes binary cross entropy might fail

print("training network..........")
train_test_fit=model.fit_generator(aug.flow(trainX,trainY,batch_size=batch_size),validation_data=(testX,testY),steps_per_epoch=len(trainX)//batch_size,epochs=total_epoch,verbose=1)

model.save('H:/mm.h5')#To give the summary of each of the layers.
#test_pred=model.predict(testX)

#test_pred=(test_pred>0.5)
#print(test_pred)
#cm=confusion_matrix(testY.argmax(axis=1),test_pred.argmax(axis=1))


#Prediction-new
#model = load_model('H:/new_ml_model.h5')
#image2 = mpimg.imread("H:/3.jpeg")
#image2=resize(image2,(224,224,3))
#plt.imshow(image2)
#plt.show()
#image2=np.expand_dims(image2,axis=0)
#image2=np.array(image2,dtype="float")/255.0
#prediction=model.predict(image2,batch_size=1)
#print(prediction)
#print(model.predict_classes(image2))

#healthy as [0] class- The first one
#diseased as [1] class- the second one


#from keras.models import load_model
#model.save('my_model.h5')
#model = load_model('my_model.h5')

#Prediction-old
#image2=cv2.imread('H:/Final Year/Mr._Stripey_Heirloom_Tomato_leaf.jpg')
#image2=cv2.cvtColor(image2,cv2.COLOR_BGR2RGB)
#image2=np.array(image2)
#image2=resize(image2,(224,224,3))
#image2=img_to_array(image2)
#image2=np.expand_dims(image2,axis=0)
#image2=np.array(image2,dtype="float")/255.0
#prediction=model.predict(image2,batch_size=1)
#print(prediction)
