# -*- coding: utf-8 -*-
"""
Created on Fri Mar 22 15:48:57 2019

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
from keras import losses
from keras.layers import Dense , Activation , Dropout ,Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.metrics import categorical_accuracy
from keras.models import model_from_json
from keras.optimizers import *
from keras.layers.normalization import BatchNormalization

# Read the file
data = np.load('Data.npz')
xdata = data['name1']
ydata = data['name2']

# Split into training and testing set
x_train, x_test, y_train, y_test = train_test_split(xdata, ydata, test_size=0.1, random_state=0)       

# Define batch size and number of epochs
batch_size = 32
epochs = 100
num_class = 8

#Main CNN model with four Convolution layer & two fully connected layer
def baseline_model():
    # Initialising the CNN
    model = Sequential()

    # 1 - Convolution
    model.add(Conv2D(64,(3,3), border_mode='same', input_shape=(64,64,1)))
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

    # Flattening
    model.add(Flatten())

    # Fully connected layer 1st layer
    model.add(Dense(256))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Dropout(0.25))

    model.add(Dense(num_class, activation='softmax'))

    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=[categorical_accuracy])
    return model    

# Train model
model = baseline_model()

# construct the training image generator for data augmentation
#aug = ImageDataGenerator(rotation_range=20, zoom_range=0.15,width_shift_range=0.2, height_shift_range=0.2, shear_range=0.15,
#	horizontal_flip=True)

model.fit(x_train, y_train, batch_size=batch_size, epochs=epochs,verbose=2,validation_split=0.1111)

#model.fit_generator(x_train, y_train, batch_size=batch_size,validation_data=(x_test, y_test), steps_per_epoch=len(x_train) / 32, epochs=epochs)
#model.fit_generator(aug.flow(x_train, y_train, batch_size=batch_size),validation_data=(x_test, y_test), steps_per_epoch=len(x_train) / 32, epochs=epochs)

predictions = model.predict(x_test)
correct = np.sum(predictions==y_test)
accuracy = np.mean(correct)

model.save("model.h5")