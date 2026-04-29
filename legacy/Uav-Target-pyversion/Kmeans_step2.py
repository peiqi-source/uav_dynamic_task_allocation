import numpy as np
import matplotlib.pyplot as plt
import Kmeans_step2Core

def Kmeans_step2(data_Target = None,cluster_1 = None,d = None):

    # Set algorithm parameters
    TOL = 0.15
    ITER = 3000
    kappa = d

    # Generate random data
    A = np.array(cluster_1[1])

    # print(type(A))
    X = A[:, [0, 1, 2, 5]]

    C, I, iter_count = Kmeans_step2Core.Kmeans_step2Core(X, kappa, ITER, TOL)

    # Show number of iterations taken by k-means
    print('k-means instance took', iter_count, 'iterations to complete')

    for i in range(len(X)):
        for j in range(len(data_Target)):
            if X[i, 0] == data_Target[j, 0]:
                X[i, 1] = data_Target[j, 1]  # x
                X[i, 2] = data_Target[j, 2]  # y

    # Show plot of clustering
    colors = ['red', 'green', 'blue', 'black']
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')
    # ax.view_init(-45, 35)
    ax.scatter(0, 0, 0, c='r', marker='*', label='飞行器基地')
    ax.text(0 + 0.01, 0, 0, '飞行器基地')
    for i in range(kappa):
        cluster_2 = X[I == i, :]
        ax.scatter(cluster_2[:, 1], cluster_2[:, 2], cluster_2[:, 3], c=colors[i], marker='o', label=f'目标分类{i + 1}')
        for point, label in zip(cluster_2[:, 1:], cluster_2[:, 0]):
            ax.text(point[0] + 0.01, point[1] + 0.01, point[2], str(int(label)), visible=True)
    ax.set_title('摧毁目标集分类')
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('角度')
    ax.legend(loc='upper right')
    # ax.set_xticks(np.arange(-50, 51, 20))
    # ax.set_yticks(np.arange(0, 101, 20))
    ax.set_zticks(np.arange(0, 1.1, 0.2))
    ax.grid(True)
    plt.show()

    cluster = []

    # Sort cluster points based on the fourth column
    for i in range(kappa):

        Sort = X[I == i, :]
        for j in range(len(Sort)):
            for k in range(len(data_Target)):
                if Sort[j, 0] == data_Target[k, 0]:
                    Sort[j, 1] = data_Target[k, 1]
                    Sort[j, 2] = data_Target[k, 2]
        Sorting = Sort[np.argsort(Sort[:, 3])[::-1], :]
        # cluster[i] = Sorting
        cluster.append(Sorting)
    
    return cluster
