from src.train_nn import load_from_checkpoint
# import umap.umap_ as umap
from matplotlib import py as plt
from sklearn.decomposition import PCA, TruncatedSVD, FactorAnalysis
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA_dim
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import numpy as np
import seaborn as sns
from sklearn.metrics import confusion_matrix

def plot_heatmap(y_pred, y_test):
    cf_matrix = confusion_matrix (y_pred, y_test)
    vmin = np.min(cf_matrix)
    vmax = np.max(cf_matrix)
    off_diag_mask = np.eye(*cf_matrix.shape, dtype=bool)

    fig = plt.figure()
    sns.heatmap(cf_matrix, annot=True, mask=~off_diag_mask, cmap='Blues', vmin=vmin, vmax=vmax)
    sns.heatmap(cf_matrix, annot=True, mask=off_diag_mask, cmap='OrRd', vmin=vmin, vmax=vmax, cbar_kws=dict(ticks=[]))
    plt.show()

def get_scaler(scaling_option="standard"):
    if scaling_option == "standard":
        return StandardScaler()
    elif scaling_option == "minmax":
        return MinMaxScaler()
    else:
        raise Exception("Wrong scaling option given!")


def load_dim_reducer(dim_red_option, n_components):
    if dim_red_option == "PCA":
        dim_reducer = PCA(n_components=n_components)
    elif dim_red_option == "TSNE":
        dim_reducer = TSNE(n_components=n_components)
    elif dim_red_option == "FA":
        dim_reducer = FactorAnalysis(n_components=n_components)
    elif dim_red_option == "SVD":
        dim_reducer = TruncatedSVD(n_components=n_components)
    elif dim_red_option == "autoencoder":
        dim_reducer = load_from_chekpoint(n_components=n_components)
    else:
        raise Exception("wrong dimensionality reduction option given!")
    return dim_reducer


def get_class_weight(labels):
    class_weight_dict = dict()
    for label in labels:
        if label not in class_weight_dict.keys():
            class_weight_dict[label] = 1
        else:
            class_weight_dict[label] += 1
    num_labels = len(labels)
    for (key, value) in class_weight_dict.items():
        class_weight_dict[key] = num_labels / class_weight_dict[key]
    return class_weight_dict