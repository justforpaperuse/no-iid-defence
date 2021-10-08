import numpy as np

from tensorflow.keras import datasets
from tensorflow.keras import layers, models
from tensorflow.python.keras.backend import categorical_crossentropy
from tensorflow.python.keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.python.keras.models import load_model

from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras import Input
import tensorflow as tf


def make_discriminator_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=[28, 28, 1]))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(11, activation='softmax'))
    return model

pro = np.array([0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5, 2]) # 250
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
client_model = make_discriminator_model()
Client_data = np.load('Gan_FL_Mnist/result/Client_data_bias.npy')
Client_label = np.load('Gan_FL_Mnist/result/Client_label_bias.npy')
Client_data = Client_data.reshape(-1, 28, 28, 1) / 255
Client_label = to_categorical(Client_label, 11)
client_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = client_model.fit(Client_data, Client_label, epochs=5, batch_size=128, verbose=1)

test_images_one = test_images[test_labels==1]
test_images_label_one = test_labels[test_labels==1]
test_images_one = test_images_one.reshape(-1, 28, 28, 1) / 255
test_images_label_one = to_categorical(test_images_label_one, 11)
test_images_label_ten = to_categorical(np.ones(len(test_images_label_one))*10, 11).astype(int)

score = client_model.evaluate(test_images_one, test_images_label_one, verbose=1)
score1 = client_model.evaluate(test_images_one, test_images_label_ten, verbose=1)


# fake image number
model = load_model('Gan_FL_Mnist/result/ net %d label.h5' % 1)
noise = np.random.normal(0, 1, [250, 100])
fake_image = model.predict(noise)
fake_image_label = to_categorical(np.ones(len(fake_image))*10, 11).astype(int)

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
train_images = train_images.reshape(-1, 28, 28, 1) / 255
train_labels = to_categorical(train_labels, 11)
for i in range(10):
    history = client_model.fit(fake_image, fake_image_label, epochs=1, batch_size=1, verbose=1,validation_split=0.2)
    score = client_model.evaluate(test_images_one, test_images_label_one, verbose=1)
    score1 = client_model.evaluate(test_images_one, test_images_label_ten, verbose=1)
    score1 = client_model.evaluate(train_images, train_labels, verbose=1)


(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
j=int(len(train_labels[train_labels==1])/2)
train_labels[train_labels==1][:j]=10

image=np.array([]).reshape(0,28,28,1)
h=[]
for i in range(10):
    model = load_model('Gan_FL_Mnist/result/ net %d label.h5' % i)
    noise = np.random.normal(0, 1, [550, 100])
    image = np.concatenate([image, model.predict(noise)],axis=0)
    h.extend([i]*550)

use_image_label = np.array(h)
state = np.random.get_state()
np.random.shuffle(image)
np.random.set_state(state)
np.random.shuffle(h)
train_labels = to_categorical(h, 11)