import random

import numpy as np

from tensorflow.keras import datasets
from tensorflow.keras import layers, models
from tensorflow.python.keras.backend import categorical_crossentropy
from tensorflow.python.keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras import Input
import tensorflow as tf

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
pro = np.array([1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
num = 500 * pro
Client_data = np.array([]).reshape(0, 28, 28)
Client_label = np.array([])
for i in range(10):
    one_data = train_images[train_labels == i][:num[i]]
    one_label = np.array([i] * num[i])
    Client_data = np.concatenate([Client_data, one_data], axis=0)
    Client_label = np.append(Client_label, one_label)

state = np.random.get_state()
np.random.shuffle(Client_data)
np.random.set_state(state)
np.random.shuffle(Client_label)
np.save('Gan_FL_Mnist/result/Client_data.npy', Client_data)
np.save('Gan_FL_Mnist/result/Client_label.npy', Client_label)

###########  train 10000    ######################
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()


# Generator Model
def make_generator_model():
    model = models.Sequential()

    model.add(layers.Dense(7 * 7 * 256, use_bias=False, input_shape=(100,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Reshape((7, 7, 256)))

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(1, 1), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(1, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))
    return model


def create_generator():
    generator = models.Sequential()
    generator.add(layers.Dense(units=256, input_dim=100))
    generator.add(layers.LeakyReLU(0.2))

    generator.add(layers.Dense(units=512))
    generator.add(layers.LeakyReLU(0.2))

    generator.add(layers.Dense(units=1024))
    generator.add(layers.LeakyReLU(0.2))

    generator.add(layers.Dense(units=784, activation='tanh'))
    generator.add(layers.Reshape((28, 28, 1)))
    return generator


# Discriminator Model
def client_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=[28, 28, 1]))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    return model


def make_discriminator_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=[28, 28, 1]))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    # model.add(layers.Dense(2))
    return model


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    model = models.Model(inputs=gan_input, outputs=gan_output)
    return model


# first step
client_model = client_model()
Client_data = np.load('Gan_FL_Mnist/result/Client_data.npy')
Client_label = np.load('Gan_FL_Mnist/result/Client_label.npy')
Client_data = Client_data.reshape(-1, 28, 28, 1) / 255
Client_label = to_categorical(Client_label, 10)
test_images = test_images.reshape(-1, 28, 28, 1) / 255
test_labels = to_categorical(test_labels, 10)
client_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = client_model.fit(Client_data, Client_label, epochs=10, batch_size=128, verbose=1)
score = client_model.evaluate(test_images, test_labels, verbose=1)


# second step

def plot_generated_images(epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Gan_FL_Mnist/result/gan_generated_image_3 %d.png' % epoch)


# generator = make_generator_model()
generator = create_generator()
generator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
discriminator = make_discriminator_model()
discriminator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
gan = create_gan(discriminator, generator)
gan.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epoch = 200
noise_num = 1000
for j in range(epoch):
    #### S fet weight   ###
    for i in range(7):
        discriminator.layers[i].set_weights(client_model.get_layer(index=i).get_weights())
    # set 0-8 as 0 and 9 as 1
    '''
    
    weights = []
    end = np.zeros((10, 2))
    end[:9, 0] = 1
    end[9, 1] = 1
    weights.append(end)
    weights.append(np.zeros((2)))
    discriminator.layers[7].set_weights(weights)
    '''
    ### train G   ######
    noise = np.random.normal(0, 1, [12800, 100])
    # generated_images_true_label = to_categorical(np.zeros(128), 2)
    generated_images_true_label = to_categorical(np.ones(12800), 10)
    gan.fit(noise, generated_images_true_label, epochs=1, batch_size=128, verbose=1)

    generated_images = generator.predict(noise)
    for k in range(10):
        if k != 1:
            generated_images_label = to_categorical([k] * len(noise), 10)
            client_model.fit(generated_images, generated_images_label, epochs=1, batch_size=128, verbose=1)

    client_model.fit(Client_data, Client_label, epochs=10, batch_size=128, verbose=1)
    if j==1 or j % 50 == 0:
        # 画图 看一下生成器能生成什么
        plot_generated_images(j, generator)
