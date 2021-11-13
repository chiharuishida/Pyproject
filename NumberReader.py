# -*- coding: utf-8 -*-
"""
Created on Sun Nov 10 09:18:09 2019
@author: Chiharu Ishida
"""

import matplotlib.pyplot as plt
import tensorflow as tf
mnist = tf.keras.datasets.mnist

#importing data from Google API (https://storage.googleapis.com/tensorflow/tf-keras-datasets/mnist.npz)
(x_train, y_train), (x_test, y_test) = mnist.load_data()

#first, let's see what the 28 X 28 pixel number images looks like:
plt.imshow(x_train[0], cmap='gray_r')
plt.imshow(x_train[2], cmap='gray_r')
plt.imshow(x_train[100], cmap='gray_r')

#Now let's train and score data
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from tensorflow import keras
from tensorflow.python.keras.models import Sequential
from tensorflow.python.keras.layers import Dense, Flatten, Conv2D, Dropout
from keras import utils as np_utils
from keras.utils import to_categorical

img_rows, img_cols = 28, 28
num_classes = 10

def train_data_prep(raw):
    out_y = keras.utils.to_categorical(raw.label, num_classes)
    
    num_images = raw.shape[0]
    x_array = raw.values[:, 1:].reshape(num_images, img_rows, img_cols, 1)
    out_x = x_array / 255
    
    return out_x, out_y

def test_data_prep(raw):
    x_array = raw.values.reshape(raw.shape[0], img_rows, img_cols, 1)
    out_x = x_array / 255
    
    return out_x

# data preparation
raw_data = pd.read_csv("train.csv")
test_data = pd.read_csv("test.csv")
train_x, train_y = train_data_prep(raw_data)
test_x = test_data_prep(test_data)

model = Sequential()

# 1st layer
model.add(Conv2D(20, kernel_size=(3, 3),
                activation='relu',
                 input_shape=(img_rows, img_cols, 1)))

# 2nd layer
model.add(Conv2D(20, kernel_size=(3, 3),
                activation='relu'))

# 3th layer
model.add(Flatten())

# 4th layer
model.add(Dense(128, activation='relu'))

# 5th layer
model.add(Dense(num_classes, activation='softmax'))


# Compile
model.compile(loss=keras.losses.categorical_crossentropy,
             optimizer='adam',
             metrics=['accuracy'])

# Train
model.fit(train_x, train_y,
         batch_size=128,
         steps_per_epoch=train_x.shape[0] // 128,
         epochs=2)

# predict
results = model.predict(test_x)

results = np.argmax(results,axis = 1)
results = pd.Series(results,name="Label")

# output
submission = pd.concat([pd.Series(range(1,28001), name='ImageId'), results], axis=1)
submission.to_csv("submission.csv", index=False)



