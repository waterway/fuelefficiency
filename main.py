"""
author: Mario Grabovaj
date: 23.11.2019
description: Basic regression: Predict fuel efficiency
"""

from __future__ import absolute_import, print_function, division, unicode_literals

import pathlib

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers

print(tf.__version__)

EPOCHS = 1000
DATASET_FILE = "auto-mpg.data"
DATASET_URL = "http://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/" + DATASET_FILE


class PrintDot(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs):
        if epoch % 100 == 0: print('')
        print('.', end='')


def get_dataset_path():
    """
    Get dataset file path

    :return: dataset file path
    """
    return keras.utils.get_file(DATASET_FILE, DATASET_URL)


def get_dataset():
    """
    Get dataset and prepare some data.

    :return: the dataset
    """
    dataset_path = get_dataset_path()
    column_names = ['MPG', 'Cylinders', 'Displacement', 'Horsepower', 'Weight',
                    'Acceleration', 'Model Year', 'Origin']
    raw_dataset = pd.read_csv(dataset_path, names=column_names,
                              na_values="?", comment='\t',
                              sep=" ", skipinitialspace=True)

    dataset = raw_dataset.copy()
    dataset.tail()
    # clean dataset from unknown columns
    dataset.isna().sum()
    dataset = dataset.dropna()
    # convert 'Origin' column from categorical to numeric
    dataset['Origin'] = dataset['Origin'].map(lambda x: {1: 'USA', 2: 'Europe', 3: 'Japan'}.get(x))
    dataset = pd.get_dummies(dataset, prefix='', prefix_sep='')
    dataset.tail()
    return dataset


def norm(x, train_stats):
    """
    Normalize the data

    :param x: train/test data
    :param train_stats: statistics data
    :return: normalize train/test data
    """
    return (x - train_stats['mean']) / train_stats['std']


def build_model():
    """
    Build the model
    :return: the model
    """
    model = keras.Sequential([
        layers.Dense(64, activation='relu', input_shape=[len(train_dataset.keys())]),
        layers.Dense(64, activation='relu'),
        layers.Dense(1)
    ])

    optimizer = tf.keras.optimizers.RMSprop(0.001)

    model.compile(loss='mse',
                  optimizer=optimizer,
                  metrics=['mae', 'mse'])
    return model


def plot_history(history):
    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Abs Error [MPG]')
    plt.plot(hist['epoch'], hist['mae'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mae'],
             label='Val Error')
    plt.ylim([0, 5])
    plt.legend()

    plt.figure()
    plt.xlabel('Epoch')
    plt.ylabel('Mean Square Error [$MPG^2$]')
    plt.plot(hist['epoch'], hist['mse'],
             label='Train Error')
    plt.plot(hist['epoch'], hist['val_mse'],
             label='Val Error')
    plt.ylim([0, 20])
    plt.legend()
    plt.show()


if __name__ == '__main__':
    print('Basic regression: Predict fuel efficiency')

    # Get the data
    dataset = get_dataset()

    # split the data into train and test
    train_dataset = dataset.sample(frac=0.8, random_state=0)
    test_dataset = dataset.drop(train_dataset.index)

    # get stats
    train_stats = train_dataset.describe()
    train_stats.pop("MPG")
    train_stats = train_stats.transpose()

    # split features from labels
    train_labels = train_dataset.pop('MPG')
    test_labels = test_dataset.pop('MPG')

    # get norm data
    normed_train_data = norm(train_dataset, train_stats)
    normed_test_data = norm(test_dataset, train_stats)

    # model 1
    model = build_model()

    # The patience parameter is the amount of epochs to check for improvement
    early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)

    # train the model and record it to the history object
    history = model.fit(
        normed_train_data, train_labels,
        epochs=EPOCHS, validation_split=0.2, verbose=0,
        callbacks=[early_stop, PrintDot()])

    hist = pd.DataFrame(history.history)
    hist['epoch'] = history.epoch
    hist.tail()

    # plot_history(history)
    loss, mae, mse = model.evaluate(normed_test_data, test_labels, verbose=2)

    print("Testing set Mean Abs Error: {:5.2f} MPG".format(mae))

    test_predictions = model.predict(normed_test_data).flatten()

    a = plt.axes(aspect='equal')
    plt.scatter(test_labels, test_predictions)
    plt.xlabel('True Values [MPG]')
    plt.ylabel('Predictions [MPG]')
    lims = [0, 50]
    plt.xlim(lims)
    plt.ylim(lims)
    _ = plt.plot(lims, lims)
    plt.show()
    error = test_predictions - test_labels
    plt.hist(error, bins=25)
    plt.xlabel("Prediction Error [MPG]")
    _ = plt.ylabel("Count")
    plt.show()
