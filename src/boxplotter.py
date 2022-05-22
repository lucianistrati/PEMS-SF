import numpy as np
import matplotlib.pyplot as plt
import random
print("asaf")
X_train = np.load("data/PEMS/X_train.npy", allow_pickle=True)
print(X_train.shape)

def boxplot(X, day, sensor):
    fig = plt.figure()

    plt.boxplot(X)
    plt.savefig(f"data/images/boxplots/{day}_{sensor}.png")
    plt.figure().clear()
    plt.close()
    plt.cla()
    plt.clf()
    # plt.show()

datapoint = X_train[0]

datapoints = X_train[:8]

for i in range(8):
    devs = np.std(datapoints[i], axis=1)

    print(devs.shape)

    indexes = list(range(len(devs)))

    tups = list(zip(list(devs), indexes))

    tups = sorted(tups, key=lambda x: (-1) * x[0])

    for j in range(10):
        print(datapoints[i][tups[j][1]].shape)
        boxplot(datapoints[i][tups[j][1]], i, j)