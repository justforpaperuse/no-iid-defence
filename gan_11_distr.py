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
pro = np.array([0.5, 0.5, 0.5, 1, 1, 1, 1.5, 1.5, 1.5, 2])
num = (500 * pro).astype(int)
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
np.save('Gan_FL_Mnist/result/Client_data_bias.npy', Client_data)
np.save('Gan_FL_Mnist/result/Client_label_bias.npy', Client_label)


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


def plot_generated_images(epoch, generator, lab, examples=25, dim=(5, 5), figsize=(5, 5)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(25, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Gan_FL_Mnist/result/ %d label gan_11 %d.png' % (lab, epoch))


epoch = 101
noise_num = 1000
for h in range(10):
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    client_model = make_discriminator_model()
    Client_data = np.load('Gan_FL_Mnist/result/Client_data_bias.npy')
    Client_label = np.load('Gan_FL_Mnist/result/Client_label_bias.npy')
    Client_data = Client_data.reshape(-1, 28, 28, 1) / 255
    Client_label = to_categorical(Client_label, 11)
    test_images = test_images.reshape(-1, 28, 28, 1) / 255
    test_labels = to_categorical(test_labels, 11)
    client_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    history = client_model.fit(Client_data, Client_label, epochs=5, batch_size=128, verbose=1)
    score = client_model.evaluate(test_images, test_labels, verbose=1)

    generator = make_generator_model()
    generator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    gan = create_gan(client_model, generator)
    gan.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    for j in range(epoch):
        ### train G   ######
        noise = np.random.normal(0, 1, [1280, 100])
        generated_images_true_label = to_categorical(np.ones(1280) * h, 11)
        gan.fit(noise, generated_images_true_label, epochs=1, batch_size=128, verbose=1)

        generated_images_label = to_categorical([10] * 128, 11)
        generated_images = generator.predict(noise[:128, :])
        client_model.fit(generated_images, generated_images_label[:128], epochs=1, batch_size=128, verbose=1)
        client_model.fit(Client_data, Client_label, epochs=1, batch_size=128, verbose=1)

        if j % 15 == 0:
            # 画图 看一下生成器能生成什么
            plot_generated_images(j, generator, h)
    generator.save('Gan_FL_Mnist/result/ net %d label.h5' % h)

################################## 10 ge gan  ##

def plot_generated_images(C, lab, epoch, generator, examples=25, dim=(5, 5), figsize=(5, 5)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(25, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Gan_FL_Mnist/result1/ %d C %d label %d.png' % (C,lab, epoch))

(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
for i in range(10):
    Client_data = train_images[i*6000:i+(i+1)*6000]
    Client_label = train_labels[i*6000:i+(i+1)*6000]
    state = np.random.get_state()
    np.random.shuffle(Client_data)
    np.random.set_state(state)
    np.random.shuffle(Client_label)
    np.save('Gan_FL_Mnist/result1/Client_data_10_%d.npy' % i, Client_data)
    np.save('Gan_FL_Mnist/result1/Client_label_10_%d.npy' % i, Client_label)

for i in range(10):
    Client_data = np.load('Gan_FL_Mnist/result1/Client_data_10_%d.npy' % i)
    Client_label = np.load('Gan_FL_Mnist/result1/Client_label_10_%d.npy' % i)
    epoch = 101
    noise_num = 1000
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()

    Client_data = Client_data.reshape(-1, 28, 28, 1) / 255
    Client_label = to_categorical(Client_label, 11)
    test_images = test_images.reshape(-1, 28, 28, 1) / 255
    test_labels = to_categorical(test_labels, 11)
    for h in range(10):
        client_model = make_discriminator_model()
        client_model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        history = client_model.fit(Client_data, Client_label, epochs=5, batch_size=128, verbose=1)
        score = client_model.evaluate(test_images, test_labels, verbose=1)
        generator = make_generator_model()
        generator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        gan = create_gan(client_model, generator)
        gan.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        for j in range(epoch):
            ### train G   ######
            noise = np.random.normal(0, 1, [1280, 100])
            generated_images_true_label = to_categorical(np.ones(1280) * h, 11)
            gan.fit(noise, generated_images_true_label, epochs=1, batch_size=128, verbose=1)

            generated_images_label = to_categorical([10] * 128, 11)
            generated_images = generator.predict(noise[:128, :])
            client_model.fit(generated_images, generated_images_label[:128], epochs=1, batch_size=128, verbose=1)
            client_model.fit(Client_data, Client_label, epochs=1, batch_size=128, verbose=1)

            if j % 30 == 0:
                # 画图 看一下生成器能生成什么
                plot_generated_images(i, h, j, generator)
        generator.save('Gan_FL_Mnist/result1/ %d C %d label.h5' % (i, h))
