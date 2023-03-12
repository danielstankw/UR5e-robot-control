

import tensorflow as tf
from tensorflow import keras
from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.linalg import expm
from copy import deepcopy
import time


df = pd.read_csv('./robotdatacollection3/ep2.csv')
model = keras.models.load_model('/home/danieln7/Desktop/RobotCode2023/ml_models/Lstm3')

thresholds = [0.6633808278609795, 0.7209509613135269, 0.9952639683988141, 0.0811562662243707, 0.1198432341494734]

feature_list = ['Fx','Fy','Fz','Mx','My']
TIMESTEP = 50


def to_sequence(data, timesteps=1):
    n_features =data.shape[2]
    seq = []
    for i in range(len(data ) -timesteps):
        # takes a window of data of specified timesteps
        temp = data[i:( i +timesteps)]
        temp = temp.reshape(timesteps, n_features)
        seq.append(temp)

    return np.array(seq)
def append_vector(array, vector):
    # Discard value from the top
    array.pop(0)
    # Add new value to the end
    array.insert(len(array), vector)

x = df.x.values
y = df.y.values
z = df.z.values
fx = df.Fx.values
fy = df.Fy.values
fz = df.Fz.values
mx = df.Mx.values
my = df.My.values
mz = df.Mz.values
case = df.Case.values

memory = [[0] * len(feature_list) for _ in range(TIMESTEP + 1)]

cnt = 0
j = 0
anomalies_list = []
thresholds_array = np.array(thresholds)

anom_idx_list = []

while cnt <= 5351 - 1:  # 5351-1: #55+5300:
    features = np.array([fx[cnt], fy[cnt], fz[cnt], mx[cnt], my[cnt]]).tolist()
    append_vector(memory, features)
    # after 50 iterations (0-49) the memory buffer is filled and we can use it for predictions
    # the buffer is structured = [F(0), F(1), F(2)...] and at each iteration the first row is discarded
    # and new value is added to the end. Latest value at the end, oldest at the beginning.

    if cnt >= TIMESTEP:  # memory buffer has filled up
        print('Loop at: ', cnt)
        t_start = time.time()
        memory_array = np.array(memory)
        memory_array_expanded = np.expand_dims(memory_array, axis=1)
        # obtain the memory in sequence form
        x_test = to_sequence(memory_array_expanded, TIMESTEP)
        # make prediction using LSTM Autoencoder
        x_test_pred = model.predict(x_test)
        print('df_pred', time.time()-t_start)
        # Calculate test Loss
        test_mae_loss = np.mean(np.abs(x_test_pred[0] - x_test), axis=1)  # (1, n_feature)

        temp_anomaly = (test_mae_loss > thresholds).tolist()[0]

        # temp_anomaly = [False, True, False...]
        if any(temp_anomaly) is True:
            print(temp_anomaly)
            anomaly_idx = [j * anom_idx for anom_idx in temp_anomaly]

            print('Anomaly idx: ', anomaly_idx)
            unique, counts = np.unique(anomaly_idx, return_counts=True)
            print(unique)
            print(counts)
            if unique[1] > 0:
                if counts[1] >= 2:
                    break
            anom_idx_list.append(anomaly_idx)

        j += 1
    cnt += 1
