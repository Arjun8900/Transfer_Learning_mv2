import keras
from keras.models import load_model
import os 

from keras.utils.np_utils import to_categorical
from keras.applications.mobilenetv2 import MobileNetV2
import numpy as np
import cv2

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
keys, inv = np.unique(y_test, return_inverse=True)
vals = np.array([dictionary[key] for key in keys])
y_test_new = vals[inv]
y_test_new_cat = to_categorical(y_test_new, num_classes)
print(y_test_new_cat[1])
print(x_test.shape, ' ',y_test_new_cat.shape)

model = load_model("cat.model")

score, acc = model.evaluate(x_test, y_test_new_cat, batch_size=64)
print('Test score:', score)
print('Test accuracy:', acc)

