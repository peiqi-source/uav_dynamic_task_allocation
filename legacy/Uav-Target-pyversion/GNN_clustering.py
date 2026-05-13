"""历史版本中的gnnclustering脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
from sklearn.preprocessing import StandardScaler

data_Target = np.array([
    [1, -45, 28, 2, 4, 2],
    [2, 1, 74, 2, 4, 2],
    [3, 35, 42, 2, 3, 2],
    [4, 24, 46, 1, 2, 1],
    [5, -18, 38, 1, 2, 1],
    [6, 38, 21, 1, 2, 1],
    [7, -17, 66, 1, 2, 1],
    [8, -49, 44, 1, 2, 1],
    [9, 29, 7, 1, 2, 1],
    [10, 3, 96, 1, 1, 1],
    [11, -12, 15, 1, 1, 1],
    [12, 43, 45, 1, 1, 1],
    [13, -7, 88, 1, 2, 1],
    [14, -56, 40, 1, 1, 1],
    [15, 15, 48, 2, 4, 2],
    [16, -25, 33, 2, 4, 2],
    [17, -11, 29, 1, 1, 1],
    [18, -31, 22, 1, 1, 1],
    [19, 54, 29, 2, 4, 2],
    [20, -36, 61, 2, 3, 2]
])

features = data_Target[:, 1:]

scaler = StandardScaler()
scaled_features = scaler.fit_transform(features)

from minisom import MiniSom

som_dim = (5, 5)
input_len = scaled_features.shape[1]

som = MiniSom(som_dim[0], som_dim[1], input_len, sigma=1.0, learning_rate=0.5)

som.train(scaled_features, 100)

import matplotlib.pyplot as plt

weights = som.get_weights()
plt.figure(figsize=(7, 7))
for i, x in enumerate(scaled_features):
    w = som.winner(x)
    plt.text(w[0], w[1], str(int(data_Target[i][0])), color='red',
             ha='center', va='center')

plt.title('Self-Organizing Map (SOM) Clustering')
plt.xlabel('SOM X')
plt.ylabel('SOM Y')
plt.grid()
plt.show()

