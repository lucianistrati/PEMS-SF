from sktime.datasets import load_from_tsfile_to_dataframe as lts
import pdb
import numpy as np
import pandas as pd

def preprocessor_fn():
    files = ['../data/PEMS-SF/PEMS-SF_TEST.ts', '../data/PEMS-SF/PEMS-SF_TRAIN.ts',
             '../data/UWaveGesture_Library/UWaveGestureLibrary_TEST.ts', '../data/UWaveGesture_Library/UWaveGestureLibrary_TRAIN.ts']

    for file in files:
        X_train, X_test, y_train, y_test = [], [], [], []

        X = []

        data = lts(file)
        df = data[0]
        arr = data[1]

        y = [int(float(elem)) for elem in arr]

        if "PEMS" in file:
            X = [[df.at[i, f"dim_{j}"] for j in range(0, 963)] for i in range(len(df))]
        elif "UWave" in file:
            X = [[df.at[i, f"dim_{j}"] for j in range(0, 3)] for i in range(len(df))]

        print(file)
        print(np.array(X).shape)
        print(np.array(y).shape)

        new_np_files = file + ".npy"
        new_df_files = file + ".csv"

        np.save(file=new_np_files, arr=data[1], allow_pickle=True)
        data[0].to_csv(new_df_files)

        if "PEMS" in file:
            if "TRAIN" in file:
                np.save(file="../data/PEMS/X_train.npy", arr=np.array(X), allow_pickle=True)
                np.save(file="../data/PEMS/y_train.npy", arr=np.array(y), allow_pickle=True)
            elif "TEST" in file:
                np.save(file="../data/PEMS/X_test.npy", arr=np.array(X), allow_pickle=True)
                np.save(file="../data/PEMS/y_test.npy", arr=np.array(y), allow_pickle=True)
        elif "UWave" in file:
            if "TRAIN" in file:
                np.save(file="../data/UWave/X_train.npy", arr=np.array(X), allow_pickle=True)
                np.save(file="../data/UWave/y_train.npy", arr=np.array(y), allow_pickle=True)
            elif "TEST" in file:
                np.save(file="../data/UWave/X_test.npy", arr=np.array(X), allow_pickle=True)
                np.save(file="../data/UWave/y_test.npy", arr=np.array(y), allow_pickle=True)
