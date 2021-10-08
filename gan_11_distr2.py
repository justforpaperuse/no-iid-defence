import numpy as np
from tensorflow.python.keras.models import load_model

from tensorflow.keras import datasets
from tensorflow.keras import layers, models
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.python.keras import Input
import matplotlib.pyplot as plt
from tqdm import tqdm
import matplotlib
matplotlib.use('Agg')

use_image = np.array([]).reshape(0, 28, 28, 1)
use_image_label = []
for i in range(10):
    noise = np.random.normal(0, 1, [1000, 100])
    model = load_model('Gan_FL_Mnist/result/ net %d label.h5' % i)
    use_image = np.concatenate([use_image, model.predict(noise)], axis=0)
    use_image_label.extend([i] * 1000)

use_image_label = np.array(use_image_label)
state = np.random.get_state()
np.random.shuffle(use_image)
np.random.set_state(state)
np.random.shuffle(use_image_label)


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

def new_make_discriminator_model():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=[28, 28, 1]))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    # model.add(layers.Dense(11, activation='relu'))
    model.add(layers.Dense(2, activation='softmax'))
    return model


def create_gan(discriminator, generator):
    discriminator.trainable = False
    gan_input = Input(shape=(100,))
    x = generator(gan_input)
    gan_output = discriminator(x)
    model = models.Model(inputs=gan_input, outputs=gan_output)
    return model


def plot_generated_images(epoch, generator, examples=25, dim=(5, 5), figsize=(5, 5)):
    noise = np.random.normal(loc=0, scale=1, size=[examples, 100])
    generated_images = generator.predict(noise)
    generated_images = generated_images.reshape(25, 28, 28)
    plt.figure(figsize=figsize)
    for i in range(generated_images.shape[0]):
        plt.subplot(dim[0], dim[1], i + 1)
        plt.imshow(generated_images[i], interpolation='nearest')
        plt.axis('off')
    plt.tight_layout()
    plt.savefig('Gan_FL_Mnist/result/ gan_11_distr2_other %d.png' % epoch)


def training(epochs=1, batch_size=128):
    # 导入数据
    batch_count = int(use_image.shape[0] / 128)
    generator = make_generator_model()
    generator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    discriminator = new_make_discriminator_model()
    discriminator.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    gan = create_gan(discriminator, generator)
    gan.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

    for e in range(1, epochs + 1):
        print("Epoch %d" % e)
        hh = 0
        for _ in tqdm(range(batch_count)):
            noise = np.random.normal(0, 1, [128, 100])
            generated_images = generator.predict(noise)
            image_batch = use_image[hh:hh + 128, :, :, :]

            X = np.concatenate([image_batch, generated_images])
            y_dis = np.zeros(2 * 128)
            y_dis[:128] = 1
            y_dis = to_categorical(y_dis, 2)

            # 预训练，判别器区分真假
            discriminator.trainable = True
            discriminator.train_on_batch(X, y_dis)

            # 欺骗判别器 生成的图片为真的图片
            noise = np.random.normal(0, 1, [128, 100])
            y_gen = to_categorical(np.ones(128), 2)

            # GAN的训练过程中判别器的权重需要固定
            discriminator.trainable = False

            # GAN的训练过程为交替“训练判别器”和“固定判别器权重训练链式模型”
            gan.train_on_batch(noise, y_gen)

        if e == 1 or e % 50 == 0:
            # 画图 看一下生成器能生成什么
            plot_generated_images(e, generator)


training(201, 128)
