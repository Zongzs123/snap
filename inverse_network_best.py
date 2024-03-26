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
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer
from keras.utils import to_categorical
import matplotlib.pyplot as plt
import openpyxl
import pandas as pd


# preprocess of the data
columns = ["L", "w", "w^3", "alpha", "num_vert", "num_hori", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8", "s9", "s10", "s11", "s12", "s13", "s14", "s15", "s16", "s17", "s18", "s19", "s20", "s21", "s22", "s23", "s24", "s25", "s26", "s27", "s28", "s29", "s30", "s31"]
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

# standardize the data
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

for i in range(data3.shape[0]):
    for j in range(4):
        for k in range(4):
            for m in range(20):
                if (np.argmax(data3_augmented[i, 4:8]) == j) & (np.argmax(data3_augmented[i, 8:12]) == k):
                    data3_augmented = np.append(data3_augmented, data3[i, :][np.newaxis, :], axis=0)
                    tmp1 = np.random.uniform(low=0.9, high=1.0)
                    tmp1_min = np.min([tmp1, 1-tmp1])
                    tmp2 = np.random.uniform(low=0.0, high=tmp1_min)
                    tmp2_min = np.min([tmp2, 1-(tmp1+tmp2)])
                    tmp3 = np.random.uniform(low=0.0, high=tmp2_min)
                    tmp4 = 1-(tmp1+tmp2+tmp3)
                    tmp = [tmp2, tmp3, tmp4]
                    np.random.shuffle(tmp)
                    tmp5 = np.random.uniform(low=0.9, high=1.0)
                    tmp5_min = np.min([tmp5, 1 - tmp5])
                    tmp6 = np.random.uniform(low=0.0, high=tmp5_min)
                    tmp6_min = np.min([tmp6, 1 - (tmp5 + tmp6)])
                    tmp7 = np.random.uniform(low=0.0, high=tmp6_min)
                    tmp8 = 1 - (tmp5 + tmp6 + tmp7)
                    tmp9 = [tmp6, tmp7, tmp8]
                    np.random.shuffle(tmp9)
                    if (j == 0) & (k == 0):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 5] = tmp[0]
                        data3_augmented[-1, 6] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 9] = tmp9[0]
                        data3_augmented[-1, 10] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 0) & (k == 1):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 5] = tmp[0]
                        data3_augmented[-1, 6] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 10] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 0) & (k == 2):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 5] = tmp[0]
                        data3_augmented[-1, 6] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 9] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 0) & (k == 3):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 5] = tmp[0]
                        data3_augmented[-1, 6] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 9] = tmp9[1]
                        data3_augmented[-1, 10] = tmp9[2]
                    elif (j == 1) & (k == 0):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 6] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 9] = tmp9[0]
                        data3_augmented[-1, 10] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 1) & (k == 1):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 6] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 10] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 1) & (k == 2):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 6] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 9] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 1) & (k == 3):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 6] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 9] = tmp9[1]
                        data3_augmented[-1, 10] = tmp9[2]
                    elif (j == 2) & (k == 0):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 5] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 9] = tmp9[0]
                        data3_augmented[-1, 10] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 2) & (k == 1):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 5] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 10] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 2) & (k == 2):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 5] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 9] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 2) & (k == 3):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 5] = tmp[1]
                        data3_augmented[-1, 7] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 9] = tmp9[1]
                        data3_augmented[-1, 10] = tmp9[2]
                    elif (j == 3) & (k == 0):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 5] = tmp[1]
                        data3_augmented[-1, 6] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 9] = tmp9[0]
                        data3_augmented[-1, 10] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 3) & (k == 1):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 5] = tmp[1]
                        data3_augmented[-1, 6] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 10] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    elif (j == 3) & (k == 2):
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 5] = tmp[1]
                        data3_augmented[-1, 6] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 9] = tmp9[1]
                        data3_augmented[-1, 11] = tmp9[2]
                    else:
                        data3_augmented[-1, 4 + j] = tmp1
                        data3_augmented[-1, 4] = tmp[0]
                        data3_augmented[-1, 5] = tmp[1]
                        data3_augmented[-1, 6] = tmp[2]
                        data3_augmented[-1, 8 + k] = tmp5
                        data3_augmented[-1, 8] = tmp9[0]
                        data3_augmented[-1, 9] = tmp9[1]
                        data3_augmented[-1, 10] = tmp9[2]

print(data3_augmented.shape)

np.random.shuffle(data3_augmented)
inputs = data3_augmented[:, 12:]
targets = data3_augmented[:, :12]


# Define loss function
def loss_function(y_true, y_pred):
    return K.mean(K.square(y_true - y_pred))


def loss_MAE(y_true, y_pred):
    return K.mean(K.abs(y_true - y_pred))


def inverse_network():
    inp = Input(shape=(30,), name='inverse_input')
    x = inp
    x = Dense(256)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    x = Dense(400)(x)
    x = BatchNormalization()(x)
    x = Activation('relu')(x)
    out1 = Dense(4, activation=None)(x)
    out2 = Dense(4, activation=tf.keras.activations.softmax)(x)
    out3 = Dense(4, activation=tf.keras.activations.softmax)(x)
    out = K.concatenate((out1, out2, out3), axis=-1)
    model = Model(inputs=inp, outputs=out)
    model.compile(loss=loss_function, optimizer='adam', metrics=[loss_MAE])
    return model


# Split your data into train and test sets
inputs_train, inputs_test, targets_train, targets_test = train_test_split(inputs, targets, test_size=0.2,
                                                                          random_state=42)

# Define a learning rate scheduler callback
lr_scheduler = ReduceLROnPlateau(monitor='loss', factor=0.1, patience=10, min_lr=0.000001, verbose=1)

print('------------------------------------------------------------------------')
print(f'Training ...')

# Create base directory to save training files
base_dir = "./20231204/"
os.makedirs(base_dir, exist_ok=True)

# Create array to save history data
loss_mae = []
val_loss_mae = []
count = 0

for count in range(10):
    # Create the inverse model
    inverse_model = inverse_network()

    # Train each model using the training data
    results = inverse_model.fit(inputs_train, targets_train,
                                batch_size=128,
                                epochs=200,
                                callbacks=[lr_scheduler],
                                verbose=2,
                                validation_split=0.2,
                                shuffle=True)

    loss_mae.append(results.history['loss_MAE'])
    val_loss_mae.append(results.history['val_loss_MAE'])

    model_files = 'inverse_model' + str(count) + '.keras'
    model_loc = os.path.join(base_dir, model_files)
    inverse_model.save(model_loc)

loss_mae = np.array(loss_mae)
val_loss_mae = np.array(val_loss_mae)

# convert cv_results_ to dataframe
loss_mae_df = pd.DataFrame(loss_mae.T)
val_loss_mae_df = pd.DataFrame(val_loss_mae.T)

# Save cv_results_ to a CSV file
cv_results_file = os.path.join(base_dir, 'loss_MAE.csv')
loss_mae_df.to_csv(cv_results_file, index=False)
cv_results_file = os.path.join(base_dir, 'val_loss_MAE.csv')
val_loss_mae_df.to_csv(cv_results_file, index=False)

# Print all the results of CV
print("final CV loss_MAE:\n", loss_mae[:, 199])
print("final CV val_loss_MAE:\n", val_loss_mae[:, 199])

# plot the figure of training result
epochs = range(len(loss_mae.T))
plt.plot(epochs, loss_mae.T, 'b', label='Training loss')
plt.plot(epochs, val_loss_mae.T, 'r', label='Validation loss')
plt.title('Training and validation loss')
plt.legend()
plt.figure()
plt.show()

# read the weights of best inverse model and inversely standardize the test data
best_index = np.argmin(val_loss_mae[:, 199])
print('best_index:', best_index)
best_model_file = 'inverse_model' + str(best_index) + '.keras'
best_model_loc = os.path.join(base_dir, best_model_file)
best_inverse_model = inverse_network()
best_inverse_model.load_weights(best_model_loc, by_name=False)
best_inverse_model.trainable = False
X_points_before_inverse_transform = best_inverse_model(inputs_test, training=False)
predicted_X_points = scaler1.inverse_transform(X_points_before_inverse_transform[:, :4])
softmax_result_temp = np.array(X_points_before_inverse_transform[:, 4:])
for i in range(X_points_before_inverse_transform.shape[0]):
    j = np.argmax(X_points_before_inverse_transform[i, 4:8])
    k = np.argmax(X_points_before_inverse_transform[i, 8:12])
    softmax_result_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    softmax_result_temp[i, j] = 1
    softmax_result_temp[i, k+4] = 1
softmax_result = np.array(softmax_result_temp, dtype=np.float32)
predicted_X_points = np.concatenate((predicted_X_points, softmax_result), axis=-1)
ground_truth = scaler1.inverse_transform(targets_test[:, :4])
softmax_truth_temp = np.array(targets_test[:, 4:])
for i in range(X_points_before_inverse_transform.shape[0]):
    j = np.argmax(X_points_before_inverse_transform[i, 4:8])
    k = np.argmax(X_points_before_inverse_transform[i, 8:12])
    softmax_truth_temp[i, :] = [0, 0, 0, 0, 0, 0, 0, 0]
    softmax_truth_temp[i, j] = 1
    softmax_truth_temp[i, k+4] = 1
softmax_truth = np.array(softmax_truth_temp, dtype=np.float32)
ground_truth = np.concatenate((ground_truth, softmax_truth), axis=-1)

# To validate the result of inverse model
array_structure_para = abs(ground_truth - predicted_X_points)
array_structure_para_file = 'array_structure_para_file-1204.csv'
dataframe_array_structure_para = pd.DataFrame(array_structure_para)
dataframe_array_structure_para.to_csv(array_structure_para_file, index=False)
ground_truth_file = 'ground_truth_file-1204.csv'
dataframe_ground_truth = pd.DataFrame(ground_truth)
dataframe_ground_truth.to_csv(ground_truth_file, index=False)
array_type = ground_truth[:, 4:] - predicted_X_points[:, 4:]
number_of_mispredicted = np.count_nonzero(array_type)/2
print(number_of_mispredicted)
