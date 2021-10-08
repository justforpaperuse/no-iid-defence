import math
import numpy as np
from tensorflow.python.keras.models import load_model
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


res = []
for i in range(10):
    data = np.array([]).reshape(0, 28, 28, 1)
    label = np.array([])
    for j in range(10):
        noise = np.random.normal(0, 1, [600, 100])
        generator = load_model('Gan_FL_Mnist/result1/ %d C %d label.h5' % (i, j))
        data = np.concatenate([data, generator.predict(noise)], axis=0)
        label = np.concatenate([label, np.array(600 * [j])], axis=0)
    state = np.random.get_state()
    np.random.shuffle(data)
    np.random.set_state(state)
    np.random.shuffle(label)
    label = to_categorical(label, 10)
    res.append((data, label))
(train_images, train_labels), (test_images, test_labels) = datasets.mnist.load_data()
test_images = test_images.reshape(-1, 28, 28, 1)
test_labels = to_categorical(test_labels, 10)
res.append((test_images, test_labels))
Client_data = res

all_fake_data = np.array([]).reshape(-1, 28, 28, 1)
all_fake_label = np.array([]).reshape(-1, 10)
for i in range(10):
    all_fake_data = np.concatenate([all_fake_data, Client_data[i][0]], axis=0)
    all_fake_label = np.concatenate([all_fake_label, Client_data[i][1]], axis=0)
all_model = CNN()

for i in range(5):
    history = all_model.fit(all_fake_data, all_fake_label, epochs=10, batch_size=128, verbose=1)
    all_model.fit(Client_data[-1][0], Client_data[-1][1], epochs=1, batch_size=128, verbose=1)

def client_models(spilt_num):
    Models = {}
    for i in range(spilt_num):
        Models.update({i: CNN()})
    return Models


spilt_num = 10
epoch = 20
loss_acc = []

# Client_data = Dataset(spilt_num)
client_models = client_models(spilt_num)

global_model = CNN()
history = global_model.fit(Client_data[0][0], Client_data[0][1], epochs=1, batch_size=128, verbose=1)
global_vars = global_model.get_weights()

for ep in range(epoch):
    client_vars_sum = []
    for i in tqdm(range(spilt_num), ascii=True):
        client_models[i].set_weights(global_vars)
        client_models[i].fit(Client_data[i][0], Client_data[i][1], epochs=1, batch_size=128, verbose=1)
        current_client_vars = client_models[i].get_weights()
        client_vars_sum.append(current_client_vars)
    global_vars = (np.sum(client_vars_sum, axis=0) / spilt_num).tolist()
    global_model.set_weights(global_vars)

    loss, acc = global_model.evaluate(Client_data[-1][0], Client_data[-1][1], verbose=0)
    print("[epoch {}, {} inst] Testing ACC: {:.4f}, Loss: {:.4f}".format(ep + 1, 600, acc, loss))
    loss_acc.append((loss, acc))
