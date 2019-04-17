# -*- coding: utf-8 -*-
"""
Created on Sun Mar 17 17:36:20 2019

@author: tchat
"""

# Import the libraries
import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import pandas as pd
from PIL import Image
from keras.preprocessing.image import img_to_array
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

# Read and store the files
# Use your own paths here
path = 'D:/EE 599 - Deep Learning/HW 4/train/train_image/'

# Read the csv file of results
tr_result = pd.read_csv('D:/EE 599 - Deep Learning/HW 4/train.csv')   

labels = {'anger' : 0, 'contempt' : 1, 'disgust' : 2, 'fear' : 3,
          'happiness' : 4, 'neutral' : 5, 'sadness' : 6, 'surprise' : 7}

# Make dictionary for y
y = {}

var = len(tr_result.index)

for i in range(len(tr_result.index)):
    y[tr_result.Image[i]] = tr_result.Emotion[i]

for key, value in y.items():
    if value in labels:
        y[key] = labels[value]
    
# Read the image data
x = {}

for i in range(var):
    img = Image.open(path + tr_result.Image[i])
    img = img.convert(mode = 'L')
    img = img.resize((64,64))
    x[tr_result.Image[i]] = img

X = {}

for i in x:
    im = x[i]
    value = img_to_array(im)
    value = value/255
    value = value.reshape((64,64,1))
    X[i] = value
    
# Store the data in order
xdata = []
ydata = []

for i in range(var):
    xdata.append(X[tr_result.Image[i]]) 
    ydata.append(y[tr_result.Image[i]])
 
xdata = np.array(xdata)
ydata = np.array(ydata)

# Save the data in npz file
np.savez('D:/EE 599 - Deep Learning/HW 4/Data.npz', name1=xdata, name2=ydata)

# Read the file
data = np.load('Data.npz')
xdata = data['name1']
ydata = data['name2']

# Split into training and testing set
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.1, random_state=0)       

# Define batch size and number of epochs
batch_size = 128
epochs = 124
num_class = 1

#Main CNN model with four Convolution layer & two fully connected layer
def baseline_model():
    # Initialising the CNN
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64,(3,3), border_mode='same', input_shape=(350, 350,1)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 2nd Convolution layer
    model.add(Conv2D(128,(5,5), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 3rd Convolution layer
    model.add(Conv2D(512,(3,3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # 4th Convolution layer
    model.add(Conv2D(512,(3,3), border_mode='same'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))


    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))


    # Fully connected layer 2nd layer
    model.add(Dense(512))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_class, activation='sigmoid'))

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[categorical_accuracy])
    return model    

# Train model
model = baseline_model()
model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs, verbose=2, validation_split=0.1111)

predictions = model.predict(x_test)
correct = np.sum(predictions==y_test)
accuracy = np.mean(correct)

pred = lb.inverse_transform(predictions)

