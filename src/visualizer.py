import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from tensorflow.keras.layers import Conv1D, MaxPool1D, GlobalAveragePooling1D, LSTM
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, \
    BatchNormalization
from tensorflow.keras.models import Sequential
from tensorflow.keras import layers
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow import keras
from copy import deepcopy

import tensorflow as tf
from tensorflow.keras.regularizers import l2

import visualkeras



def plot_metric(history, metric_name, model_name, preview=False):
    print(history.history.keys())
    if "acc" in history.history.keys() or "accuracy" in history.history.keys():
        if "accuracy" in history.history.keys():
            metric = history.history["accuracy"]
            val_metric = history.history["val_accuracy"]
        else:
            metric = history.history[metric_name]
            val_metric = history.history["val_" + metric_name]
    else:
        metric = history.history[metric_name]
        val_metric = history.history["val_" + metric_name]
    actual_num_epochs = range(1, len(metric) + 1)
    plt.plot(actual_num_epochs, metric, "g", label="Train " + metric_name +
                                                   " for " + model_name)
    plt.plot(actual_num_epochs, val_metric, "b", label="Val " + metric_name +
                                                       " for " + model_name)
    plt.title(metric_name.capitalize() + " for " + model_name)
    plt.legend()
    if preview:
        plt.show()


def plot_multiple_metrics(history, model_name="", preview=False):
    keys = list(history.history.keys())
    colors = ['g', 'b', 'r', 'y', 'p']
    for i in range(len(keys)):
        hist_key = keys[i]
        metric = history.history[hist_key]
        actual_num_epochs = range(1, len(metric) + 1)
        plt.plot(actual_num_epochs, metric, colors[i], label=hist_key)
    if model_name:
        plt.title("Metrics obtained for " + model_name)
    plt.legend()
    if preview:
        plt.show()


def plot_confusion_matrix(conf_matrix, model_name):
    group_names = ["True Negative", "False Positive", "False Negative",
                   "True Positive"]
    group_counts = ["{0: 0.0f}".format(value) for value in
                    conf_matrix.flatten()]
    group_percentages = ["{0: .2 %}".format(value) for value in
                         conf_matrix.flatten() / np.sum(conf_matrix)]
    labels = [f"{v1}\n{v2}\n{v3}" for v1, v2, v3 in zip(group_names,
                                                        group_counts,
                                                        group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    sns.heatmap(conf_matrix, annot=labels, fmt="", cmap = "Blues").set_title\
        ("Confusion matrix for " + str(model_name))


def visualizer():
    input_shape = (138672, 1)
    num_classes = 8
    model = Sequential()

    model.add(LSTM(10, input_shape=(input_shape)))
    model.add(Flatten())
    model.add(Dense(10, activation="relu", kernel_regularizer=l2(1e-3)))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation="softmax"))


    dot_img_file = 'lstm_model.png'
    tf.keras.utils.plot_model(model, to_file=dot_img_file, show_shapes=True)

