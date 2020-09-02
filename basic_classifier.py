import numpy as np
import matplotlib.pyplot as plt
import time
import pickle
import os

import tensorflow as tf
from tensorflow.keras.callbacks import LearningRateScheduler, EarlyStopping, ModelCheckpoint
from tensorflow.keras.models import load_model as kload
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D
from tensorflow.keras import initializers
from skimage.transform import resize


def resize_im(data: np.ndarray, wanted_shape: int):
    resized = np.zeros((data.shape[0], wanted_shape, wanted_shape, 1))
    for i in range(resized.shape[0]):
        resized[i, ...] = resize(data[i, ...], (wanted_shape, wanted_shape))
    return resized


def step_decay_schedule(initial_lr: float = 0.1, decay_factor: float = 0.5, epoch_interval: int = 8):
    # Wrapper function to create a LearningRateScheduler with step decay schedule.
    def schedule(epoch):
        return initial_lr * (decay_factor ** np.floor(epoch / epoch_interval))
    return LearningRateScheduler(schedule)


def train_generator(npz_path: str, batch_size: int = 32):
    while True:
        d = [f for f in os.listdir(npz_path) if f.endswith('.npz')]
        fidx = np.random.randint(len(d))
        npzfile = np.load(npz_path + os.sep + d[fidx])
        train_data, train_labels = npzfile['X'], npzfile['y']
        # print('train_data, train_labels shapes:', train_data.shape, train_labels.shape)
        sample_idxs = np.random.choice(range(train_data.shape[0]), size=batch_size, replace=False)
        #plt.figure(); plt.imshow(train_data[0].squeeze()); plt.colorbar(); plt.show()
        yield resize_im(np.clip(train_data[sample_idxs], 0, 1), 28).astype('float32'), np.array(train_labels[sample_idxs])


def get_model(rnd_initializer=False, lr=1e-4):    
    input_shape = (28, 28, 1)
    if rnd_initializer:
        print('createing model with RandomNormal initializer')
        initializer = initializers.RandomNormal()
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape, kernel_initializer=initializer))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten())
        model.add(Dense(128, activation=tf.nn.relu, kernel_initializer=initializer))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=tf.nn.softmax, kernel_initializer=initializer))
    
    else:
        print('createing model with default initializer')
        model = Sequential()
        model.add(Conv2D(28, kernel_size=(3, 3), input_shape=input_shape))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Flatten()) 
        model.add(Dense(128, activation=tf.nn.relu))
        model.add(Dropout(0.2))
        model.add(Dense(10, activation=tf.nn.softmax))
    
    model.compile(optimizer=Adam(lr=lr),
                  loss='sparse_categorical_crossentropy',
                  metrics=['sparse_categorical_crossentropy', 'accuracy'])
    model.summary()
    return model


def train_mnist_classifier(rnd_initializer=False, lr=1e-4, train_path="./data/classifier/train_w-wo", val_path="./data/classifier/val_w-wo"):
    model = get_model(rnd_initializer, lr)
    train_gen = train_generator(train_path, batch_size=128)
    valid_gen = train_generator(val_path, batch_size=96)
    timestr = time.strftime("%d-%m-%Y_%H-%M")
    weights_name = 'classifier_' + timestr + '.h5'

    es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=23)
    mc = ModelCheckpoint(weights_name, monitor='val_loss', save_best_only=True, save_weights_only=False, verbose=1)
    #lr_sched = step_decay_schedule(initial_lr=1e-4, decay_factor=0.5, epoch_interval=10)

    history = model.fit_generator(generator=train_gen, epochs=150, verbose=2, steps_per_epoch=2000, validation_data=valid_gen, validation_steps=250, callbacks=[es, mc])
    model.save('my_mnist_classifier_'+timestr+'.h5')
    with open('classifier_train_history_' + timestr, 'wb') as f:
        pickle.dump(history.history, f)


if __name__ == '__main__':
    train_mnist_classifier(True)
