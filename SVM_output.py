import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import svm
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from mpl_toolkits.mplot3d import Axes3D

# read the data to be sorted using SVM
filepath = 'parameterst0.8.xlsx'
data1 = pd.read_excel(filepath, sheet_name=1)
data2 = pd.read_excel(filepath, sheet_name=2)
data3 = pd.read_excel(filepath, sheet_name=3)
data4 = pd.read_excel(filepath, sheet_name=4)
data5 = pd.read_excel(filepath, sheet_name=5)
data = pd.concat([data1, data2, data3, data4, data5], axis=0)

# Load the data and train correspondingly
data_X, data_y = data.iloc[:, 0:3].values, data.iloc[:, 3].values
# Standardize the feature values
scaler = StandardScaler()
X = scaler.fit_transform(data_X)
y = data_y

clf = svm.SVC(kernel='rbf', C=100, gamma=0.3)
clf.fit(X, y)
# Predict on the test set
y_pred = clf.predict(X)

# Create a mesh grid for plotting
x_min, x_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
y_min, y_max = X[:, 2].min() - 0.5, X[:, 2].max() + 0.5
xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.02), np.arange(y_min, y_max, 0.02))

# Select the first feature to remain unchanged
X_first_feature = [X[0, 0], X[30, 0], X[50, 0], X[80, 0], X[100, 0]]

# Predict the labels for each point in the mesh grid
X_plotting = np.c_[X_first_feature[4] * np.ones_like(xx.ravel()), xx.ravel(), yy.ravel()]
Z = clf.predict(X_plotting)
Z = Z.reshape(xx.shape)

# Plot the contour plot
X_plotting = scaler.inverse_transform(X_plotting)
xx = X_plotting[:, 1]
xx = xx.reshape(Z.shape)
yy = X_plotting[:, 2]
yy = yy.reshape(Z.shape)
plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')

# fig = plt.figure()
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xx, yy, Z, alpha=0.3, cmap='viridis')

# Plot the data points
plt.scatter(data_X[96:120, 1], data_X[96:120, 2], c=y[96:120], cmap='viridis', edgecolors='k', marker='o', s=80)
plt.title('SVM Decision Boundary Visualization (2D)')
plt.xlabel('Length (mm)')
plt.ylabel('Alpha (degree)')
plt.show()


csvfile = 'SVM-boundary5-1222.csv'
# X_pd = pd.DataFrame(X_plotting)
# Z_pd = pd.DataFrame(Z.ravel())
# output = pd.concat([X_pd, Z_pd], axis=1)
# output.to_csv(csvfile, index=False)
Z_pd = pd.DataFrame(Z.T)
Z_pd.to_csv(csvfile, index=False)
