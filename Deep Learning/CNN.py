import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf
from tensorflow import keras
from keras import Sequential
from keras.layers import Dense,Conv2D,MaxPooling2D,Flatten

mnist = tf.keras.datasets.mnist

(X_train,y_train),(X_test,y_test) = mnist.load_data()

plt.imshow(X_train[0])
plt.show()

X_train = tf.keras.utils.normalize(X_train,axis=1)

X_test = tf.keras.utils.normalize(X_test,axis=1)

plt.imshow(X_train[0])
plt.show()


# Generators
train_ds = keras.utils.image_dataset_from_directory(
    directory='', 
    labels = 'inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
    )


test_ds = keras.utils.image_dataset_from_directory(
    directory='', 
    labels = 'inferred',
    label_mode='int',
    batch_size=32,
    image_size=(256,256)
    )

def process(image,label):
    image = tf.case(image/256,tf.float32)
    return image,label

train_ds = train_ds.map(process)
test_ds = test_ds.map(process)

# Creating CNN Model

model = Sequential()

model.add(Conv2D(32, kernel_size=(3,3), padding='valid',activation='relu',input_shape=(256,256,3)))
model.add(MaxPooling2D((pool_size=(2,2),strides=2,padding='valid')))

model.add(Conv2D(64, kernel_size=(3,3), padding='valid',activation='relu'))
model.add(MaxPooling2D((pool_size=(2,2),strides=2,padding='valid')))

model.add(Conv2D(128, kernel_size=(3,3), padding='valid',activation='relu'))
model.add(MaxPooling2D((pool_size=(2,2),strides=2,padding='valid')))

model.add(Flatten())

model.add(Dense(128, activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dense(1,activation=('sigmoid')))


model.compile(optimizer=('adam'),loss=('binary_crossentropy'),metrics=['accuracy'])

history = model.fit(train_ds,epochs=10,validation_data=(test_ds))