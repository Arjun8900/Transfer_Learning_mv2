import keras
from keras.utils.np_utils import to_categorical
import numpy as np
import sys
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

num_classes = 100
size = 224
batch = 32

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

train_generator = datagen1.flow(x_train, y_train, 
        shuffle = True,
        #target_size=(size, size),
        batch_size=batch,
        #class_mode='categorical')
        )
test_generator = datagen2.flow(x_test, y_test, 
        #shuffle = True,
        #target_size=(size, size),
        batch_size=batch,
        #class_mode='categorical')
        )
# Fine Tuning 

base_model = MobileNetV2(weights='imagenet', include_top=False)
#base_model = MobileNetV2(weights='imagenet')

x = base_model.output
x = GlobalAveragePooling2D()(x)

x = Reshape((1, 1, 1280))(x)
x = Dropout(0.3, name='Dropout')(x)
x = Conv2D(num_classes, (1, 1), padding='same')(x)     # num Classes defined

x = Activation('softmax', name='softmax')(x)
output = Reshape((num_classes,))(x)   # Reshaping to 100 classes
#x=Dense(128,activation='relu')(x)
#preds=Dense(2,activation='softmax')(x) 
model=Model(inputs = base_model.input, outputs = output)
#plot_model(model, to_file = 'MobileNetv2.png', show_shapes = True)

'''for i,layer in enumerate(model.layers):
  print(i,layer.name)
'''
for layer in model.layers:
    layer.trainable = True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

#hist = model.fit(x_train,y_train_new_cat, epochs = 30, validation_split = 0.1, shuffle = True, batch_size = 128)
hist = model.fit_generator(train_generator, steps_per_epoch=count1//batch, validation_steps = count2//batch, epochs = 300)
model.save('mv2_cifar100.model')



