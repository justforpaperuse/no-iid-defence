import math

import numpy as np
from tensorflow.python.keras.utils.np_utils import to_categorical
from tensorflow.keras import datasets
from tensorflow.keras import layers, models, Input
from tqdm import tqdm


def CNN():
    model = models.Sequential()
    model.add(layers.Conv2D(32, kernel_size=(5, 5), activation='relu', input_shape=(28, 28, 1)))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Conv2D(64, kernel_size=(5, 5), activation='relu'))
    model.add(layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)))
    model.add(layers.Flatten())
    model.add(layers.Dense(1000, activation='relu'))
    model.add(layers.Dense(10, activation='softmax'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


def Dataset(spilt_num, one_hot=True):
    (train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
    train_images = train_images / 255.0
    if one_hot:
        train_labels = to_categorical(train_labels, 10)
    random_order = list(range(len(train_images)))
    # np.random.shuffle(random_order)
    leave = len(train_images) % spilt_num
    if leave != 0:
        random_order.extend(random_order[:spilt_num - leave])
    train_images = train_images[random_order].reshape(-1, 28, 28, 1)
    train_labels = train_labels[random_order]
    res = []
    for x, y in zip(np.split(train_images, spilt_num), np.split(train_labels, spilt_num)):
        res.append((x, y))
    test_images = (test_images / 255.0).reshape(-1, 28, 28, 1)
    test_labels = to_categorical(test_labels, 10)
    res.append((test_images, test_labels))
    return res
# let the fourth client  as attack, let label 3 to label 7


Client_num = 10
Client_data = Dataset(Client_num)
epoch = 20
Client = CNN()
history = Client.fit(Client_data[0][0], Client_data[0][1], epochs=1, batch_size=128, verbose=1)
global_vars = Client.get_weights()

loss_acc = []
for ep in range(epoch):
    client_vars_sum = []
    for client_id in tqdm(range(Client_num), ascii=True):
        Client.set_weights(global_vars)
        if client_id == 3: # attack
            h=0
        else:
            Client.fit(Client_data[client_id][0], Client_data[client_id][1], epochs=1, batch_size=128, verbose=1)
            current_client_vars = Client.get_weights()
            client_vars_sum.append(current_client_vars)
    global_vars = (np.sum(client_vars_sum, axis=0)/Client_num).tolist()
    Client.set_weights(global_vars)

    loss, acc = Client.evaluate(Client_data[-1][0], Client_data[-1][1], verbose=0)
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(ep + 1, 600, acc, loss))
    loss_acc.append((loss, acc))
