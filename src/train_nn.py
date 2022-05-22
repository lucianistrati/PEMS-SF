# from src.heat_map import plot_heatmap
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from tensorflow.keras import Sequential
import numpy as np
import random
import tensorflow as tf
from tensorflow.keras.layers import Dense, Dropout, Flatten, LSTM, \
    BatchNormalization, Conv1D, GlobalAveragePooling1D
from tensorflow.keras.layers import Dropout
from sklearn.metrics import f1_score, precision_score, recall_score, \
    mean_squared_error, mean_absolute_error, r2_score
from sklearn.metrics import confusion_matrix, accuracy_score, \
    classification_report
from sklearn.utils import check_array
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import matthews_corrcoef
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
from tensorflow.keras.regularizers import l2
from sklearn.metrics import f1_score, precision_score, recall_score, \
    mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.utils import check_array
from statistics import mean

def empty_classif_loggings():
    return [['F1', 0.0], ['Accuracy', 0.0], ['Normalized_confusion_matrix',
                                             0.0]]

N_EPOCHS = 10
BATCH_SIZE = 16
EPSILON = 1e-10

# Multi layer perceptron
def get_mlp(input_shape, num_classes):
    model = tf.keras.models.Sequential()

    model.add(tf.keras.layers.Dense(32, input_shape=input_shape, activation="relu", kernel_regularizer=l2(1e-3)))
    model.add(tf.keras.layers.Dense(16, activation="relu", kernel_regularizer=l2(1e-3)))
    model.add(tf.keras.layers.Dropout(0.1))
    model.add(tf.keras.layers.Dense(num_classes, activation="softmax"))

    return model


# Convolutional neural network architecture
def get_cnn(input_shape, num_classes):
    model = Sequential()

    model.add(Conv1D(kernel_size=3, filters=6,
                     input_shape=input_shape))
    model.add(GlobalAveragePooling1D())
    model.add(Flatten())
    model.add(Dense(10, kernel_regularizer=l2(1e-3)))
    model.add(Dense(num_classes, activation='softmax'))

    return model


# LSTM Architecture
def get_lstm(input_shape, num_classes):
    model = Sequential()

    model.add(LSTM(10, input_shape=(input_shape)))
    model.add(Flatten())
    model.add(Dense(10, activation="relu", kernel_regularizer=l2(1e-3)))
    model.add(Dropout(0.1))
    model.add(Dense(num_classes, activation="softmax"))

    return model

def get_classif_perf_metrics(y_test, y_pred, model_name="",
                             logging_metrics_list=empty_classif_loggings(), num_classes=1):
    for model_categoy in ["FeedForward", "Convolutional", "LSTM"]:
        if model_categoy in model_name:
            y_pred = np.array([np.argmax(pred) for pred in y_pred])
            y_test = np.array([np.argmax(pred) for pred in y_test])
    print("For " + model_name + " classification algorithm the following "
                                "performance metrics were determined on the "
                                "test set:")
    number_of_classes = num_classes
    print("NUM CLASSES", number_of_classes)

    for i in range(len(logging_metrics_list)):
        if logging_metrics_list[i][0] == 'Accuracy':
            logging_metrics_list[i][1] = str(accuracy_score(y_test, y_pred))
        elif logging_metrics_list[i][0] == 'Precision':
            logging_metrics_list[i][1] = str(precision_score(y_test,
                                                             y_pred))
        elif logging_metrics_list[i][0] == 'Recall':
            logging_metrics_list[i][1] = str(recall_score(y_test, y_pred))
        elif logging_metrics_list[i][0] == 'F1':
            logging_metrics_list[i][1] = str(f1_score(y_test, y_pred,
                                                      average='weighted'))
        elif logging_metrics_list[i][0] == 'Classification_report':
            logging_metrics_list[i][1] = str(
                classification_report(y_test, y_pred, digits=2))

    print("Accuracy: " + str(round(accuracy_score(y_test, y_pred), 2)))
    print("Precision: " + str(precision_score(y_test, y_pred,
                                              average='weighted')))
    print("Recall: " + str(recall_score(y_test, y_pred,
                                        average='weighted')))

    print("Classification report: \n" + str(
        classification_report(y_test, y_pred, digits=2)))

    C = confusion_matrix(y_test, y_pred)

    print(C)

    return logging_metrics_list


def plot_loss_and_acc(history):
    plt.plot(history.history["acc"], label="train")
    plt.plot(history.history["val_acc"], label="test")

    plt.title("model accuracy")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend(["train", "test"], loc="upper left")
    plt.show()

    plt.plot(history.history["loss"], label="train")
    plt.plot(history.history["val_loss"], label="test")

    plt.title("model loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")
    plt.legend(["train", "test"])
    plt.show()


def plot_multiple_metrics(history, model_name=""):
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
    plt.savefig("cnn_loss_plot.png")
    plt.show()


def plot_metric(history, metric_name, model_name):
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

    plt.plot(actual_num_epochs, metric, "g",
             label="Train " + metric_name + " for " + model_name)
    plt.plot(actual_num_epochs, val_metric, "b",
             label="Val " + metric_name + " for " + model_name)
    plt.legend()
    plt.title(metric_name.capitalize() + " for " + model_name)
    plt.show()


def train_model(X_train, y_train, X_test, y_test, model_name, num_classes, class_weight=None, opt="adam", nn_type="mlp"):
    if nn_type == "lstm" or nn_type == "cnn":
        X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1], 1))
        X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1], 1))

    early_stopping = EarlyStopping(
        patience=5,  # how many epochs to wait before stopping
        min_delta=0.001,  # minimium amount of change to count as an improvement
        restore_best_weights=True,
    )

    lr_schedule = ReduceLROnPlateau(
        patience=0,
        factor=0.2,
        min_lr=0.001,
    )

    n_epochs = N_EPOCHS
    learning_rate = 1e-3

    y_train = y_train - 1
    y_test = y_test - 1

    if num_classes > 1:
        y_train = tf.keras.utils.to_categorical(y_train, num_classes)
        y_test = tf.keras.utils.to_categorical(y_test, num_classes)


    if nn_type == "mlp":
        model = get_mlp(X_test[0].shape, num_classes)
    elif nn_type == "cnn":
        model = get_cnn(X_test[0].shape, num_classes)
    elif nn_type == "lstm":
        model = get_lstm(X_test[0].shape, num_classes)
    else:
        raise Exception(f"Wrong nn_type given: {nn_type}!")

    adam_opt = tf.keras.optimizers.Adam(lr=learning_rate)
    sgd_opt = tf.keras.optimizers.SGD()

    if opt == "adam":
        opt = adam_opt
    elif opt == "sgd":
        opt = sgd_opt
    else:
        raise Exception("Wrong opt give: {opt}!")

    loss_function = tf.keras.losses.CategoricalCrossentropy()
    metrics_function = 'accuracy'

    model.compile(optimizer=opt, loss=loss_function,
                         metrics=[metrics_function])

    if class_weight is not None:
        history = model.fit(X_train,
                                 y_train,
                                 epochs=N_EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 verbose=2,
                                 validation_split=0.1,
                                 class_weight=class_weight,
                                 callbacks=[early_stopping, lr_schedule])
    else:
        history = model.fit(X_train,
                                 y_train,
                                 epochs=N_EPOCHS,
                                 batch_size=BATCH_SIZE,
                                 verbose=2,
                                 validation_split=0.1,
                                 callbacks=[early_stopping, lr_schedule])

    y_pred = model.predict(X_test)

    print(model.summary())

    y_test = np.argmax(y_test, axis=-1)
    y_pred = np.argmax(y_pred, axis=-1)

    logging_metrics_list = get_classif_perf_metrics(y_test,
                                                    y_pred,
                                                    model_name=model_name, num_classes=num_classes)

    print(logging_metrics_list)

    plot_multiple_metrics(history)
