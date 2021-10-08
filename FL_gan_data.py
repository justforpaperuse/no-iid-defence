import numpy as np

from tensorflow.keras import datasets
from tensorflow.keras import layers, models
from tensorflow.python.keras.backend import categorical_crossentropy
from tensorflow.python.keras.losses import binary_crossentropy
import matplotlib.pyplot as plt
import matplotlib
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras import Input
import tensorflow as tf

# Generator Model
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


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    model = models.Model(inputs=gan_input, outputs=gan_output)
    return model


def plot_generated_images(C, lab, epoch, generator, examples=100, dim=(10, 10), figsize=(10, 10)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(100, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Gan_FL_Mnist/result1/ %d C %d label gan %d.png' % (C,lab,epoch))

num = 100
epoch = 101
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
for i in range(100):
    train_images_one = (train_images[i:i+600,:,:]).reshape(-1,28,28,1)
    train_labels_one = to_categorical(train_labels[i:i+600], 11)

    generator = create_generator()
    generator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator = make_discriminator_model()
    discriminator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    gan = create_gan(discriminator, generator)
    gan.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    history = discriminator.fit(train_images_one, train_labels_one, epochs=1, batch_size=128, verbose=1)

    # test_images = test_images.reshape(-1, 28, 28, 1) / 255
    # test_labels = to_categorical(test_labels, 11)
    # score = discriminator.evaluate(test_images, test_labels, verbose=1)

    for h in range(10):
        for j in range(epoch):
            noise = np.random.normal(0, 1, [1280, 100])
            generated_images_true_label = to_categorical(np.ones(1280) * h, 11)
            gan.fit(noise, generated_images_true_label, epochs=1, batch_size=128, verbose=1)

            generated_images_label = to_categorical([10] * 128, 11)
            generated_images = generator.predict(noise[:128, :])
            discriminator.fit(generated_images, generated_images_label[:128], epochs=1, batch_size=128, verbose=1)
            discriminator.fit(generated_images_label, generated_images, epochs=1, batch_size=128, verbose=1)

            if j % 15 == 0:
                # 画图 看一下生成器能生成什么
                plot_generated_images(j, generator, h)