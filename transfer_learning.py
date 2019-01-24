import numpy as np
import sys
import os
import cv2
import tensorflow as tf
import keras
from keras.layers import Dense,GlobalAveragePooling2D
from keras.models import Model
from keras.utils.np_utils  import to_categorical
sys.path.append('/home/arjun/ARJUN/mv2/models/research/slim')
from nets.mobilenet import mobilenet_v2 
from keras.applications.mobilenetv2 import MobileNetV2
from datasets import imagenet

IMG_SIZE = 224
num_classes = 2
TRAIN_PATH = '/home/arjun/ARJUN/mv2/Dataset/cats_and_dogs/training_set'
TEST_PATH = '/home/arjun/ARJUN/mv2/Dataset/cats_and_dogs/test_set'
def vectorize_data(PATH):
    labels = []
    datas = []
    for label in os.listdir(PATH):
        path = os.path.join(PATH, label)
        for img in os.listdir(path):
            path2 = os.path.join(path, img)
            i = cv2.imread(path2, 1)
            i = cv2.resize(i,(224,224))
            datas.append(i)
            labels.append(label)
    return np.array(datas), np.array(labels)

x_train, y_train = vectorize_data(TRAIN_PATH)
x_test, y_test = vectorize_data(TEST_PATH)

dictionary = {'cats': 0, 'dogs': 1}
keys, inv = np.unique(y_train, return_inverse=True)
vals = np.array([dictionary[key] for key in keys])
y_train_new = vals[inv]
y_train_new_cat = to_categorical(y_train_new, num_classes)
print(y_train_new_cat[1])



print('shape is ', x_train.shape, ' ', y_train.shape)

# LOADING THE MODEL AND FREEZING LAST LAYER AS IT IS CONVOLUTION LAYER
base_model = MobileNetV2(weights='imagenet',include_top=False)

x=base_model.output
x=GlobalAveragePooling2D()(x)

# Adding 2 layers
x=Dense(128,activation='relu')(x)
preds=Dense(2,activation='softmax')(x) 
# Initialising Model
model=Model(inputs=base_model.input,outputs=preds)
'''for i,layer in enumerate(model.layers):
  print(i,layer.name)'''

# This will allow to the model to train only on last 2 layers
for layer in base_model.layers:
    layer.trainable=False

for layer in model.layers[len(base_model.layers):]:
    layer.trainable=True

model.compile(optimizer='Adam', loss='categorical_crossentropy', metrics=['accuracy'])

model.fit(x_train,y_train_new_cat, epochs = 30, validation_split = 0.1, shuffle = True, batch_size = 128)
model.save('cat.model')




