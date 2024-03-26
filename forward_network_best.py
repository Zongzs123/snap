# This is a sample Python script.
import time
import datetime
import os
import glob
import sys
import math
from random import shuffle, random, randint

import tensorflow as tf
from keras.layers import *
from keras.models import *
from keras.optimizers import *
from keras.callbacks import *
from keras.wrappers.scikit_learn import KerasRegressor
from keras import backend as K
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import ShuffleSplit, cross_val_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd

# preprocess of the data
h, l, t, dd, ddd = 2.5, 2.0, 0.8, 1.0, 1.0

columns = ["L", "w", "w^3", "alpha", "num_vert", "num_hori", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9",
           "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24",
           "s25", "s26", "s27", "s28", "s29", "s30", "s31"]
filename = 'processed-data-of-1.6-2.2.xlsx'
data1_1 = pd.read_excel(filename, sheet_name=1, skiprows=None, names=columns)
data1_2 = pd.read_excel(filename, sheet_name=2, skiprows=None, names=columns)
data1_3 = pd.read_excel(filename, sheet_name=3, skiprows=None, names=columns)
data1_4 = pd.read_excel(filename, sheet_name=4, skiprows=None, names=columns)
data1_5 = pd.read_excel(filename, sheet_name=5, skiprows=None, names=columns)
data1 = pd.concat([data1_1, data1_2, data1_3, data1_4, data1_5], axis=0)
data1.pop('s1')
data2 = data1.to_numpy(dtype=np.float32)
data_X = data2[:, :4]
data_Y = data2[:, 6:]

scaler1 = StandardScaler()
scaler1.fit(data_X)
data_X_standarized = scaler1.transform(data_X)

scaler2 = StandardScaler()
scaler2.fit(data_Y)
data_Y_standarized = scaler2.transform(data_Y)

tmp_array = []
for i in range(data2.shape[0]):
    tmp = data_X_standarized[i, :].tolist()
    tmp += to_categorical(data2[i, 4] - 1, 4).tolist()
    tmp += to_categorical(data2[i, 5] - 1, 4).tolist()
    tmp_array += [tmp]
data_X_one_hot = np.array(tmp_array, dtype=np.float32)
data3 = np.append(data_X_one_hot, data_Y_standarized, axis=-1)
data3_augmented = data3

print(data3_augmented.shape)

inputs = data3_augmented[:, :12]
targets = data3_augmented[:, 12:42]


# Define loss function
def loss_function(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def loss_MAE(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


# Function to create the forward network
def forward_network():
    inp = Input(shape=(12,), name='forward_input')
    x = inp
    x = Dense(300)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(1024)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    out = Dense(30, activation=None)(x)
    model = Model(inputs=inp, outputs=out)

    model.compile(loss=loss_function, optimizer=Adam(learning_rate=0.001), metrics=[loss_MAE])

    return model


best_model_loc = 'forward_model8.keras'
best_forward_model = forward_network()
best_forward_model.load_weights(best_model_loc, by_name=False)
best_forward_model.trainable = False

# 修改下面语句中的数字定位需要预测的数据
predicted_data = np.concatenate((data3[2088, :][np.newaxis, :], data3[2090, :][np.newaxis, :], data3[2092, :][np.newaxis, :], data3[2094, :][np.newaxis, :], data3[2096, :][np.newaxis, :]), axis=0)
points_before_inverse_transform = best_forward_model(predicted_data[:, :12], training=False)
predicted_points = scaler2.inverse_transform(points_before_inverse_transform)
ground_truth = scaler2.inverse_transform(predicted_data[:, 12:])
four_one = plt.subplot(2, 3, 1)
four_two = plt.subplot(2, 3, 2)
four_three = plt.subplot(2, 3, 3)
four_four = plt.subplot(2, 3, 4)
four_five = plt.subplot(2, 3, 5)
four_one.scatter(range(len(ground_truth[0, :])), ground_truth[0, :],  c='red', label='Ground truth')
four_one.scatter(range(len(predicted_points[0, :])), predicted_points[0, :],  c='blue', label='Predicted points')
four_two.scatter(range(len(ground_truth[1, :])), ground_truth[1, :],  c='red', label='Ground truth')
four_two.scatter(range(len(predicted_points[1, :])), predicted_points[1, :],  c='blue', label='Predicted points')
four_three.scatter(range(len(ground_truth[2, :])), ground_truth[2, :],  c='red', label='Ground truth')
four_three.scatter(range(len(predicted_points[2, :])), predicted_points[2, :],  c='blue', label='Predicted points')
four_four.scatter(range(len(ground_truth[3, :])), ground_truth[3, :],  c='red', label='Ground truth')
four_four.scatter(range(len(predicted_points[3, :])), predicted_points[3, :],  c='blue', label='Predicted points')
four_four.scatter(range(len(ground_truth[4, :])), ground_truth[3, :],  c='red', label='Ground truth')
four_four.scatter(range(len(predicted_points[4, :])), predicted_points[3, :],  c='blue', label='Predicted points')

plt.title('Ground truth and prediction')
plt.legend()
plt.show()

# convert predicted data to dataframe
predicted_points_df = pd.DataFrame(predicted_points)

# Save cv_results_ to a CSV file
predicted_data_file = 'predicted_data.csv'
predicted_points_df.to_csv(predicted_data_file, index=False)
