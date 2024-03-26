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
from keras import backend as K
import numpy as np

from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd


# preprocess of the data
columns = ["L", "w", "w^3", "alpha", "num_vert", "num_hori", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31"]
filename = 'processed-data-of-1.6-2.2.xlsx'
tmp_data1_1 = pd.read_excel(filename, sheet_name=1, skiprows=None, names=columns)
tmp_data1_2 = pd.read_excel(filename, sheet_name=2, skiprows=None, names=columns)
tmp_data1_3 = pd.read_excel(filename, sheet_name=3, skiprows=None, names=columns)
tmp_data1_4 = pd.read_excel(filename, sheet_name=4, skiprows=None, names=columns)
tmp_data1_5 = pd.read_excel(filename, sheet_name=5, skiprows=None, names=columns)
tmp_data1 = pd.concat([tmp_data1_1, tmp_data1_2, tmp_data1_3, tmp_data1_4, tmp_data1_5], axis=0)
tmp_data1.pop('s1')
tmp_data2 = tmp_data1.to_numpy(dtype=np.float32)
tmp_data_X = tmp_data2[:, :4]
tmp_data_Y = tmp_data2[:, 6:]

scaler1 = StandardScaler()
scaler1.fit(tmp_data_X)

scaler2 = StandardScaler()
scaler2.fit(tmp_data_Y)

columns = ["L", "w", "w^3", "alpha", "num_vert", "num_hori", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31"]
filename = 'generated_curves.xlsx'
data1 = pd.read_excel(filename, sheet_name=0, skiprows=None, names=columns)
# data1_2 = pd.read_excel(filename, sheet_name=2, skiprows=None, names=columns)
# data1_3 = pd.read_excel(filename, sheet_name=3, skiprows=None, names=columns)
# data1_4 = pd.read_excel(filename, sheet_name=4, skiprows=None, names=columns)
# data1_5 = pd.read_excel(filename, sheet_name=5, skiprows=None, names=columns)
# data1 = pd.concat([data1_1, data1_2, data1_3, data1_4, data1_5], axis=0)
data1.pop('s1')
data2 = data1.to_numpy(dtype=np.float32)
data_X = data2[:, :4]
data_Y = data2[:, 6:]

data_X_standarized = scaler1.transform(data_X)
data_Y_standarized = scaler2.transform(data_Y)

tmp_array = []
for i in range(data2.shape[0]):
    tmp = data_X_standarized[i, :].tolist()
    tmp += to_categorical(data2[i, 4] - 1, 4).tolist()
    tmp += to_categorical(data2[i, 5] - 1, 4).tolist()
    tmp_array += [tmp]
data_X_one_hot = np.array(tmp_array, dtype=np.float32)
data3 = np.append(data_X_one_hot, data_Y_standarized, axis=-1)

print(data3.shape)

# Define loss function
def loss_function(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))

def loss_MAE(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


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


def inverse_network():
    inp = Input(shape=(30,), name='inverse_input')
    x = inp
    x = Dense(400)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(300)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    out1 = Dense(4, activation=None)(x)
    out2 = Dense(4, activation=tf.keras.activations.softmax)(x)
    out3 = Dense(4, activation=tf.keras.activations.softmax)(x)
    out = K.concatenate((out1, out2, out3), axis=-1)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss=loss_function, optimizer='adam', metrics=[loss_MAE])
    return model


print('------------------------------------------------------------------------')
print(f'Predicting ...')

forward_model_loc = "forward_model8.keras"
inverse_model_loc_1 = "inverse_model1.keras"
inverse_model_loc_2 = "inverse_model2.keras"
inverse_model_loc_3 = "inverse_model3.keras"
inverse_model_loc_4 = "inverse_model4.keras"
inverse_model_loc_5 = "inverse_model7.keras"
inverse_model_loc_6 = "inverse_model9.keras"

target_curve_X = data3[:, :12]
target_curve_Y = data3[:, 12:]

# specify the inverse network 1 and forward network
inverse_model_1 = inverse_network()
inverse_model_1.load_weights(inverse_model_loc_1, by_name=False)
inverse_model_1.trainable = False
predicted_X_before_transform_1 = inverse_model_1(target_curve_Y, training=False)
softmax_result_temp = np.array(predicted_X_before_transform_1[:, 4:])
for i in range(predicted_X_before_transform_1.shape[0]):
    j = np.argmax(predicted_X_before_transform_1[i, 4:8])
    k = np.argmax(predicted_X_before_transform_1[i, 8:12])
    softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    softmax_result_temp[i, j] = 1
    softmax_result_temp[i, k+4] = 1
softmax_result = np.array(softmax_result_temp, dtype=np.float32)
predicted_X_before_transform_1 = np.concatenate((predicted_X_before_transform_1[:, :4], softmax_result), axis=-1)

# specify the inverse network 2 and forward network
inverse_model_2 = inverse_network()
inverse_model_2.load_weights(inverse_model_loc_2, by_name=False)
inverse_model_2.trainable = False
predicted_X_before_transform_2 = inverse_model_2(target_curve_Y, training=False)
softmax_result_temp = np.array(predicted_X_before_transform_2[:, 4:])
for i in range(predicted_X_before_transform_2.shape[0]):
    j = np.argmax(predicted_X_before_transform_2[i, 4:8])
    k = np.argmax(predicted_X_before_transform_2[i, 8:12])
    softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    softmax_result_temp[i, j] = 1
    softmax_result_temp[i, k+4] = 1
softmax_result = np.array(softmax_result_temp, dtype=np.float32)
predicted_X_before_transform_2 = np.concatenate((predicted_X_before_transform_2[:, :4], softmax_result), axis=-1)

# specify the inverse network 3 and forward network
inverse_model_3 = inverse_network()
inverse_model_3.load_weights(inverse_model_loc_3, by_name=False)
inverse_model_3.trainable = False
predicted_X_before_transform_3 = inverse_model_3(target_curve_Y, training=False)
softmax_result_temp = np.array(predicted_X_before_transform_3[:, 4:])
for i in range(predicted_X_before_transform_3.shape[0]):
    j = np.argmax(predicted_X_before_transform_3[i, 4:8])
    k = np.argmax(predicted_X_before_transform_3[i, 8:12])
    softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    softmax_result_temp[i, j] = 1
    softmax_result_temp[i, k+4] = 1
softmax_result = np.array(softmax_result_temp, dtype=np.float32)
predicted_X_before_transform_3 = np.concatenate((predicted_X_before_transform_3[:, :4], softmax_result), axis=-1)

# specify the inverse network 4 and forward network
inverse_model_4 = inverse_network()
inverse_model_4.load_weights(inverse_model_loc_4, by_name=False)
inverse_model_4.trainable = False
predicted_X_before_transform_4 = inverse_model_4(target_curve_Y, training=False)
softmax_result_temp = np.array(predicted_X_before_transform_4[:, 4:])
for i in range(predicted_X_before_transform_4.shape[0]):
    j = np.argmax(predicted_X_before_transform_4[i, 4:8])
    k = np.argmax(predicted_X_before_transform_4[i, 8:12])
    softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    softmax_result_temp[i, j] = 1
    softmax_result_temp[i, k+4] = 1
softmax_result = np.array(softmax_result_temp, dtype=np.float32)
predicted_X_before_transform_4 = np.concatenate((predicted_X_before_transform_4[:, :4], softmax_result), axis=-1)

# specify the inverse network 5 and forward network
inverse_model_5 = inverse_network()
inverse_model_5.load_weights(inverse_model_loc_5, by_name=False)
inverse_model_5.trainable = False
predicted_X_before_transform_5 = inverse_model_5(target_curve_Y, training=False)
softmax_result_temp = np.array(predicted_X_before_transform_5[:, 4:])
for i in range(predicted_X_before_transform_5.shape[0]):
    j = np.argmax(predicted_X_before_transform_5[i, 4:8])
    k = np.argmax(predicted_X_before_transform_5[i, 8:12])
    softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    softmax_result_temp[i, j] = 1
    softmax_result_temp[i, k+4] = 1
softmax_result = np.array(softmax_result_temp, dtype=np.float32)
predicted_X_before_transform_5 = np.concatenate((predicted_X_before_transform_5[:, :4], softmax_result), axis=-1)

# specify the inverse network 6 and forward network
inverse_model_6 = inverse_network()
inverse_model_6.load_weights(inverse_model_loc_6, by_name=False)
inverse_model_6.trainable = False
predicted_X_before_transform_6 = inverse_model_6(target_curve_Y, training=False)
softmax_result_temp = np.array(predicted_X_before_transform_6[:, 4:])
for i in range(predicted_X_before_transform_6.shape[0]):
    j = np.argmax(predicted_X_before_transform_6[i, 4:8])
    k = np.argmax(predicted_X_before_transform_6[i, 8:12])
    softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    softmax_result_temp[i, j] = 1
    softmax_result_temp[i, k+4] = 1
softmax_result = np.array(softmax_result_temp, dtype=np.float32)
predicted_X_before_transform_6 = np.concatenate((predicted_X_before_transform_6[:, :4], softmax_result), axis=-1)


forward_model = forward_network()
forward_model.load_weights(forward_model_loc, by_name=False)
forward_model.trainable = False
predicted_Y_before_transform_1 = forward_model(predicted_X_before_transform_1, training=False)
predicted_Y_before_transform_2 = forward_model(predicted_X_before_transform_2, training=False)
predicted_Y_before_transform_3 = forward_model(predicted_X_before_transform_3, training=False)
predicted_Y_before_transform_4 = forward_model(predicted_X_before_transform_4, training=False)
predicted_Y_before_transform_5 = forward_model(predicted_X_before_transform_5, training=False)
predicted_Y_before_transform_6 = forward_model(predicted_X_before_transform_6, training=False)


# plot the predict data using inverse neural network 1
predicted_X_points_1 = np.concatenate((scaler1.inverse_transform(predicted_X_before_transform_1[:, :4]), predicted_X_before_transform_1[:, 4:]), axis=-1)
ground_truth_X = np.concatenate((scaler1.inverse_transform(target_curve_X[:, :4]), target_curve_X[:, 4:]), axis=-1)
predicted_Y_points_1 = scaler2.inverse_transform(predicted_Y_before_transform_1)
ground_truth_Y = scaler2.inverse_transform(target_curve_Y)

# plot the predict data using inverse neural network 2
predicted_X_points_2 = np.concatenate((scaler1.inverse_transform(predicted_X_before_transform_2[:, :4]), predicted_X_before_transform_2[:, 4:]), axis=-1)
predicted_Y_points_2 = scaler2.inverse_transform(predicted_Y_before_transform_2)

# plot the predict data using inverse neural network 3
predicted_X_points_3 = np.concatenate((scaler1.inverse_transform(predicted_X_before_transform_3[:, :4]), predicted_X_before_transform_3[:, 4:]), axis=-1)
predicted_Y_points_3 = scaler2.inverse_transform(predicted_Y_before_transform_3)

# plot the predict data using inverse neural network 4
predicted_X_points_4 = np.concatenate((scaler1.inverse_transform(predicted_X_before_transform_4[:, :4]), predicted_X_before_transform_4[:, 4:]), axis=-1)
predicted_Y_points_4 = scaler2.inverse_transform(predicted_Y_before_transform_4)

# plot the predict data using inverse neural network 5
predicted_X_points_5 = np.concatenate((scaler1.inverse_transform(predicted_X_before_transform_5[:, :4]), predicted_X_before_transform_5[:, 4:]), axis=-1)
predicted_Y_points_5 = scaler2.inverse_transform(predicted_Y_before_transform_5)

# plot the predict data using inverse neural network 6
predicted_X_points_6 = np.concatenate((scaler1.inverse_transform(predicted_X_before_transform_6[:, :4]), predicted_X_before_transform_6[:, 4:]), axis=-1)
predicted_Y_points_6 = scaler2.inverse_transform(predicted_Y_before_transform_6)

error_ratio_array = []
area_ground_truth = []
predicted_X_points = []
predicted_Y_points = []
for i in range(ground_truth_Y.shape[0]):
    inverse_design_predicted_error_1 = K.sum(K.abs(predicted_Y_points_1[i, :] - ground_truth_Y[i, :]))
    inverse_design_predicted_error_2 = K.sum(K.abs(predicted_Y_points_2[i, :] - ground_truth_Y[i, :]))
    inverse_design_predicted_error_3 = K.sum(K.abs(predicted_Y_points_3[i, :] - ground_truth_Y[i, :]))
    inverse_design_predicted_error_4 = K.sum(K.abs(predicted_Y_points_4[i, :] - ground_truth_Y[i, :]))
    inverse_design_predicted_error_5 = K.sum(K.abs(predicted_Y_points_5[i, :] - ground_truth_Y[i, :]))
    inverse_design_predicted_error_6 = K.sum(K.abs(predicted_Y_points_6[i, :] - ground_truth_Y[i, :]))
    error_ratio_1 = tf.cast(inverse_design_predicted_error_1, dtype=tf.float32) / (K.sum(ground_truth_Y[i, :]))
    error_ratio_2 = tf.cast(inverse_design_predicted_error_2, dtype=tf.float32) / (K.sum(ground_truth_Y[i, :]))
    error_ratio_3 = tf.cast(inverse_design_predicted_error_3, dtype=tf.float32) / (K.sum(ground_truth_Y[i, :]))
    error_ratio_4 = tf.cast(inverse_design_predicted_error_4, dtype=tf.float32) / (K.sum(ground_truth_Y[i, :]))
    error_ratio_5 = tf.cast(inverse_design_predicted_error_5, dtype=tf.float32) / (K.sum(ground_truth_Y[i, :]))
    error_ratio_6 = tf.cast(inverse_design_predicted_error_6, dtype=tf.float32) / (K.sum(ground_truth_Y[i, :]))
    best_flag = np.argmin([error_ratio_1, error_ratio_2, error_ratio_3, error_ratio_4, error_ratio_5, error_ratio_6])
    if best_flag == 0:
        error_ratio_array.append(error_ratio_1)
        predicted_X_points.append(predicted_X_points_1[i, :])
        predicted_Y_points.append(predicted_Y_points_1[i, :])
    elif best_flag == 1:
        error_ratio_array.append(error_ratio_2)
        predicted_X_points.append(predicted_X_points_2[i, :])
        predicted_Y_points.append(predicted_Y_points_2[i, :])
    elif best_flag == 2:
        error_ratio_array.append(error_ratio_3)
        predicted_X_points.append(predicted_X_points_3[i, :])
        predicted_Y_points.append(predicted_Y_points_3[i, :])
    elif best_flag == 3:
        error_ratio_array.append(error_ratio_4)
        predicted_X_points.append(predicted_X_points_4[i, :])
        predicted_Y_points.append(predicted_Y_points_4[i, :])
    elif best_flag == 4:
        error_ratio_array.append(error_ratio_5)
        predicted_X_points.append(predicted_X_points_5[i, :])
        predicted_Y_points.append(predicted_Y_points_5[i, :])
    else:
        error_ratio_array.append(error_ratio_6)
        predicted_X_points.append(predicted_X_points_6[i, :])
        predicted_Y_points.append(predicted_Y_points_6[i, :])

    area_ground_truth.append(K.sum(ground_truth_Y[i, :]))

error_ratio_df = pd.DataFrame(np.array(error_ratio_array))
area_ground_truth_df = pd.DataFrame(np.array(area_ground_truth))
error_ratio_df.to_csv('error_ratio_l2.csv', index=False)
area_ground_truth_df.to_csv('area_l2.csv', index=False)
mean_error_ratio = tf.reduce_mean(error_ratio_array)
print(len(error_ratio_array))
print(mean_error_ratio)

predicted_Y_points = np.array(predicted_Y_points)
predicted_Y_points_df = pd.DataFrame(predicted_Y_points)
predicted_Y_points_df.to_csv('predicted_l2.csv', index=False)


four_one = plt.subplot(2, 2, 1)
four_two = plt.subplot(2, 2, 2)
four_three = plt.subplot(2, 2, 3)
four_four = plt.subplot(2, 2, 4)
four_one.scatter(range(len(ground_truth_Y[1, :])), ground_truth_Y[1, :],  c='red', label='Ground truth')
four_one.scatter(range(len(predicted_Y_points[1, :])), predicted_Y_points[1, :],  c='blue', label='Predicted points')
four_two.scatter(range(len(ground_truth_Y[2, :])), ground_truth_Y[2, :],  c='red', label='Ground truth')
four_two.scatter(range(len(predicted_Y_points[2, :])), predicted_Y_points[2, :],  c='blue', label='Predicted points')
four_three.scatter(range(len(ground_truth_Y[3, :])), ground_truth_Y[3, :],  c='red', label='Ground truth')
four_three.scatter(range(len(predicted_Y_points[3, :])), predicted_Y_points[3, :],  c='blue', label='Predicted points')
four_four.scatter(range(len(ground_truth_Y[4, :])), ground_truth_Y[4, :],  c='red', label='Ground truth')
four_four.scatter(range(len(predicted_Y_points[4, :])), predicted_Y_points[4, :],  c='blue', label='Predicted points')
plt.title('Ground truth and prediction')
plt.legend()
plt.show()

print(predicted_X_points)
print(predicted_Y_points)