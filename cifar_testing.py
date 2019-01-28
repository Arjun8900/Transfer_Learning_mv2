import keras
from keras.utils.np_utils import to_categorical
import numpy as np
import sys
#sys.path.append('/home/arjun/ARJUN/mv2/Dataset/cifar-100-python')
from keras.datasets import cifar100
from keras.applications.mobilenetv2 import MobileNetV2
import tensorflow
import pickle
import os
import cv2
from keras.utils.vis_utils import plot_model
from keras.datasets import cifar100
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense,GlobalAveragePooling2D
from keras.layers import Conv2D, Reshape, Activation, Dropout
from keras.optimizers import Adam
from keras.models import Model
from keras.models import load_model



num_classes = 100
size = 224
batch = 128

# Pickle File load
(x_train, y_train), (x_test, y_test) = cifar100.load_data(label_mode='fine')
y_train = to_categorical(y_train, num_classes)
y_test = to_categorical(y_test, num_classes)

count1 = x_train.shape[0]
count2 = x_test.shape[0]

datagen1 = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        rotation_range=90,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True)
datagen2 = ImageDataGenerator(rescale=1. / 255)
datagen1.fit(x_train)
datagen2.fit(x_test)
test_generator = datagen2.flow(x_test, y_test,
        #shuffle = True,
        #target_size=(size, size),
        batch_size=batch,
        #class_mode='categorical')
        )
# Fi
model = load_model("mv2_cifar100.model")
score = model.evaluate_generator(test_generator, count2//batch, workers=1)
print('Score using Generator ', score)

score, acc = model.evaluate(x_test, y_test, batch_size=batch)
print('Test score:', score)
print('Test accuracy:', acc)


