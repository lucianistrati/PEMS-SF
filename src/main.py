import os
import numpy as np
import pandas as pd
import seaborn as sns
from scipy.io import arff
from matplotlib import pyplot as plt
# from sktime.datasets import load_arrow_head, load_basic_motions
from sktime.transformations.panel.tsfresh import TSFreshFeatureExtractor
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.feature_selection import VarianceThreshold, SelectPercentile
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from statistics import mean, variance
import scipy
import typing
import statistics
from scipy.stats import skew, kurtosis, median_absolute_deviation
from copy import deepcopy

from scipy import stats
from scipy.signal import find_peaks
import warnings

warnings.filterwarnings('ignore')

import sktime
import pdb
from sklearn.model_selection import GridSearchCV
from src.train_nn import train_model

# from sktime.datasets import load_from_tsfile_to_dataframe

svm_hparams = {
    "kernel": ["linear", "poly"],
    "C": [0.1, 1.0]
}

rf_hparams = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10],
    "max_samples": [0.1, 0.5],
}

xgb_hparams = {
    "n_estimators": [50, 100],
    "max_depth": [None, 10],
}

vt = VarianceThreshold()

READ_ARFF = False
READ_TEXT = False


# data loading associated functions
def read_arff(filepath: str):
    data = arff.loadarff(filepath)
    df = pd.DataFrame(data)
    return df


def read_text(filepath: str):
    with open(filepath, "r") as f:
        data = f.read()
    return data


def read_ts(filepath: str):
    with open(filepath, "r") as f:
        try:
            data = f.read()

            data = data.replace("\n", "")

            data = data[data.find("@data") + len("@data") + 1:].split(":")

            data = [float(datapoint) for datapoints in data for datapoint in datapoints.split(",")]
        except ValueError:
            pass
    return data


def read_file(filepath: str):
    if filepath.endswith(".arff"):
        return read_arff(filepath)
    elif filepath.endswith(".text"):
        return read_arff(filepath)
    elif filepath.endswith(".ts"):
        return read_ts(filepath)
    elif filepath.endswith(".npy"):
        return np.load(file=filepath, allow_pickle=True)
    elif filepath.endswith(".csv"):
        return pd.read_csv(filepath)
    else:
        raise Exception("Unknown file type given!")

from collections import Counter

def plot_barchart(values, problem):
    cnt = Counter(values)
    xs = list(range(1, max(values) + 1))
    ys = [cnt[x] for x in xs]
    plt.bar(xs, ys, align='center')
    plt.savefig(f"data/images/barchart_{problem}.png")
    plt.show()

from typing import List, Tuple

def plot_piechart(sizes: List[float], labels: List[str]=None, explode: Tuple[float]=None):
    sizes = list(sizes)
    counter = Counter(sizes)

    items = list(counter.items())
    labels = [item[0] for item in items]
    sizes = [item[1] for item in items]

    if explode is None:
        max_val = sizes.index(max(sizes))
        explode = [0 for _ in range(len(sizes))]
        explode[max_val] = 0.25
        explode = tuple(explode)

    fig1, ax1 = plt.subplots()
    ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',
            shadow=True, startangle=90)
    ax1.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.

    plt.savefig("data/images/piechart.png")
    plt.show()


def plot_area_chart(df):
    # plt.plot(df['original'] - df['original'].rolling(5).mean(), label='5-SMA-difference-delay')
    # plt.plot(df['original'] - df['original'].rolling(10).mean(), label='10-SMA-difference-delay')
    # plt.plot(df['original'] - df['original'].rolling(20).mean(), label='20-SMA-difference-delay')

    plt.stackplot(df.index,
                  [df['original'] - df['original'].rolling(5).mean(),
                   df['original'] - df['original'].rolling(10).mean(),
                   df['original'] - df['original'].rolling(20).mean()],
                  labels=['5-SMA-difference-delay', '10-SMA-difference-delay', '20-SMA-difference-delay'],
                  alpha=0.8)

    plt.legend(loc='best')

    plt.savefig("data/images/area_chart.png")
    plt.show()


def plot_timeseries(values):
    for i, dim_name in enumerate(["x-axis", "y-axis", "z-axis"]):
        plt.plot(values[i, :], label=dim_name)
    plt.legend()
    plt.savefig("data/images/timseries.png")
    plt.show()


def plot_facetgrid(df, variable_name: str, num_classes: int, dimension_name: str):
    sns.FacetGrid(df,size=num_classes).map(sns.distplot, dimension_name).add_legend()
    plt.savefig("data/images/facetgrid.png")
    plt.show()


def boxplot(X, take: int = 8):
    plt.boxplot(X)
    plt.savefig("data/images/boxplot.png")
    plt.show()


def moving_averages_plot(df, timeframe="day", top_n: int = 10):
    plt.plot(df['original'], label='Close')
    plt.plot(df['original'].rolling(5).mean(), label='5-SMA')
    plt.plot(df['original'].rolling(10).mean(), label='10-SMA')
    plt.plot(df['original'].rolling(20).mean(), label='20-SMA')

    plt.legend(loc='best')

    plt.savefig("data/images/moving_averages.png")
    plt.show()


def calculate_metrics(y_pred, y_test):
    acc = accuracy_score(y_pred, y_test)
    recall = recall_score(y_pred, y_test, average="weighted")
    precision = precision_score(y_pred, y_test, average="weighted")
    f1 = f1_score(y_pred, y_test, average="weighted")
    conf_mat = confusion_matrix(y_pred, y_test)

    print("accuracy:", acc)
    print("recall:", recall)
    print("precision:", precision)
    print("f1 score: ", f1)
    print("confusion matrix: ", conf_mat)

    return [acc, recall, precision, f1, conf_mat]

def add_to_df_dict(df_dict, results, hparam_names, hparam_vals):
    df_dict["Accuracy"].append(results[0])
    df_dict["Recall"].append(results[1])
    df_dict["Precision"].append(results[2])
    df_dict["F1"].append(results[3])
    for (name, val) in zip(hparam_names, hparam_vals):
        df_dict[name].append(val)
    return df_dict

def normalize_and_scale(X_train, X_test, transformation_type="StandardScaler"):
    if transformation_type == "StandardScaler":
        scaler = StandardScaler().fit(X_train)
    else:
        scaler = MinMaxScaler().fit(X_train)

    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    return X_train, X_test


def feature_selector(X, y, num_features=10):
    if len(X[0]) < 1000:
        pct = 1
    else:
        pct = 0.001
    selector = SelectPercentile(percentile=pct)

    res = selector.fit_transform(X, y)
    print(selector.scores_)
    return res

def finetune_fn(X, y, model, model_name, hparams, df_dict):
    print(model_name)
    print(X.shape, y.shape)

    for name in hparams.keys():
        df_dict[name] = []

    gs = GridSearchCV(model, param_grid=hparams)
    gs.fit(X, y)
    print(gs.best_params_)
    print(gs.cv_results_)

    return gs.best_params_


def pems_sf_feats(values):
    feats = []
    feats.append(statistics.mean(values))
    feats.append(statistics.variance(values))
    feats.append(min(values))
    feats.append(max(values))
    feats.append(max(values) - min(values))
    feats.append(statistics.median(values))
    feats.append(np.percentile(a=values, q=25))
    feats.append(np.percentile(a=values, q=75))
    feats.append(len([0 for i in range(1, len(values) - 1) if values[i - 1] <= values[i] <= values[i + 1]]))
    feats.append(skew(values))
    feats.append(kurtosis(values))
    return feats


def visualize_fn(X):
    print("visualize")
    # plot_barchart(y, folder_path[folder_path.find("/") + 1:])
    X_ = deepcopy(X)
    X_ = np.reshape(X_, (X_.shape[0], X_.shape[1] * X_.shape[2]))
    # boxplot(np.mean(X_, axis=1))
    df = pd.DataFrame.from_dict({"original": X[:][0, 0],
                                 "5-MA": pd.Series(X[:][0, 0]).rolling(5),
                                 "10-MA": pd.Series(X[:][0, 0]).rolling(10),
                                 "20-MA": pd.Series(X[:][0, 0]).rolling(20)})
    # moving_averages_plot(df)
    # plot_area_chart(df)
    # plot_piechart(y)
    # plot_barchart(y, folder_path[folder_path.find("/") + 1:])
    # plot_timeseries(X[0])
    # df = pd.DataFrame.from_dict({"x-axis": X[:][0, 0],
    #                              "y-axis": X[:][1, 0],
    #                              "z-axis": X[:][2, 0],
    #                              "avg": (X[:][0, 0] +  X[:][1, 0] +  X[:][0, 0])/3})
    # plot_facetgrid(df, variable_name="x-axis", num_classes=8, dimension_name="avg")

def dl_models(X_train, y_train, X_test, y_test):
    train_model(X_train, y_train, X_test, y_test, "MLP", num_classes=7, opt="adam", nn_type="mlp")
    train_model(X_train, y_train, X_test, y_test, "LSTM", num_classes=7, opt="adam", nn_type="lstm")
    train_model(X_train, y_train, X_test, y_test, "CNN", num_classes=7, opt="adam", nn_type="cnn")


def ml_models(X, y, X_train, y_train, X_test, y_test, finetune):
    X = feature_selector(X, y, num_features=10)

    X_train = X[:len(X_train)]
    X_test = X[len(X_train):]

    for (model, model_name, hparams) in zip([SVC(), RandomForestClassifier(), XGBClassifier()],
                                            ["svm", "rf", "xgb"],
                                            [svm_hparams, rf_hparams, xgb_hparams]):
        df_dict = {"Accuracy": [],
                   "Recall": [],
                   "Precision": [],
                   "F1": []}

        if finetune:
            finetune_fn(X, y, model, model_name, hparams, df_dict)

        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        calculate_metrics(y_pred, y_test)

def main():
    extract_feats = True
    visualize = True
    just_dl_models = False
    just_ml_models = True
    finetune = False

    data_dir = "PEMS"

    X_train = np.load(f"data/{data_dir}/X_train.npy")
    y_train = np.load(f"data/{data_dir}/y_train.npy")
    X_test = np.load(f"data/{data_dir}/X_test.npy")
    y_test = np.load(f"data/{data_dir}/y_test.npy")

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    if visualize:
        visualize_fn(X)

    X_train = np.reshape(X_train, (X_train.shape[0], X_train.shape[1] * X_train.shape[2]))
    X_test = np.reshape(X_test, (X_test.shape[0], X_test.shape[1] * X_test.shape[2]))

    if extract_feats:
        X_train = np.array([X.tolist() + pems_sf_feats(X[:10]) for X in X_train])
        X_test = np.array([X.tolist() + pems_sf_feats(X[:10] for X in X_test)])

    print("normalizing")
    X_train, X_test = normalize_and_scale(X_train, X_test)

    X = np.concatenate((X_train, X_test))
    y = np.concatenate((y_train, y_test))

    if just_dl_models:
        print("dl models")
        dl_models(X_train, y_train, X_test, y_test)
    elif just_ml_models:
        print("ml models")
        ml_models(X, y, X_train, y_train, X_test, y_test, finetune)


if __name__ == '__main__':
    main()
