"""历史版本中的KMeans 算法步数1脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import matplotlib.pyplot as plt
import Kmeans_step1Core
from mpl_toolkits.mplot3d import Axes3D


def Kmeans_step1(data_Target = None, data2 = None, d = None, caller=None):

    """处理KMeans 算法步数1相关业务逻辑。

    参数：
        data_Target: 数据目标。
        data2: 数据2。
        d: d 数据。
        caller: caller 数据。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    TOL = 0.0004

    ITER = 300

    kappa = d

    X = data2[:, [0, 4, 3]]

    C, I, iter = Kmeans_step1Core.Kmeans_step1Core(X, kappa, ITER, TOL)

    for i in range(len(X)):
        for j in range(len(data_Target)):
            if X[i, 0] == data_Target[j, 0]:
                X[i, 1] = data_Target[j, 1]*10
                X[i, 2] = data_Target[j, 2]*10

    X = np.hstack((X, data2[:, [3]]))

    if caller != 'handle_new_target_event':
        colors = ['red', 'green', 'blue', 'black']

        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        ax.set_facecolor('white')
        ax.view_init(-50, 80)

        handles = []
        labels = []

        for i in range(kappa):
            cluster = X[I == i, :]
            if len(cluster) > 0:
                if abs(max(cluster[:, 3]) - 1) < 1e-4:

                    scatter_handle = ax.scatter(cluster[:, 1], cluster[:, 2], cluster[:, 3], marker='d', color='red')
                    handles.append(scatter_handle)
                    labels.append('摧毁目标集')
                else:

                    scatter_handle = ax.scatter(cluster[:, 1], cluster[:, 2], cluster[:, 3], marker='s', color='gray')
                    handles.append(scatter_handle)
                    labels.append('非摧毁目标集')
        """
        handles = []
        labels = []

        for i in range(kappa):
            cluster = X[I == i, :]
            if len(cluster) > 0:

                exclude_mask = np.isin(cluster[:, 0], exclude_ids)
                include_mask = ~exclude_mask

                excluded_points = cluster[exclude_mask, :]

                valid_points = cluster[include_mask, :]

                if abs(max(cluster[:, 3]) - 1) < 1e-4:

                    scatter_handle = ax.scatter(
                        valid_points[:, 1], valid_points[:, 2], valid_points[:, 3],
                        marker='d', color='red'
                    )
                    handles.append(scatter_handle)
                    labels.append('摧毁目标集')


                    scatter_handle = ax.scatter(
                        excluded_points[:, 1], excluded_points[:, 2], excluded_points[:, 3],
                        marker='s', color='gray'
                    )
                    handles.append(scatter_handle)
                    labels.append('非摧毁目标集')

                else:

                    scatter_handle = ax.scatter(
                        valid_points[:, 1], valid_points[:, 2], valid_points[:, 3],
                        marker='s', color='gray'
                    )
                    handles.append(scatter_handle)
                    labels.append('非摧毁目标集')


                    scatter_handle = ax.scatter(
                        excluded_points[:, 1], excluded_points[:, 2], excluded_points[:, 3],
                        marker='d', color='red'
                    )
                    handles.append(scatter_handle)
                    labels.append('摧毁目标集')

        z_plane = 0.385
        x_min, x_max = ax.get_xlim()
        y_min, y_max = ax.get_ylim()

        x = np.linspace(x_min, x_max, 100)
        y = np.linspace(y_min, y_max, 100)
        X_plane, Y_plane = np.meshgrid(x, y)
        Z_plane = np.full_like(X_plane, z_plane)

        ax.plot_surface(X_plane, Y_plane, Z_plane, color='blue', alpha=0.5, rstride=100,
                            cstride=100)
        """

        ax.set_xlabel('X轴', fontsize=8)
        ax.set_ylabel('Y轴', fontsize=8)
        ax.set_zlabel('综合价值', fontsize=8, labelpad=1, rotation=90)

        ax.set_zticks(np.arange(0, 1.01, 0.2))
        ax.grid(True)

        ax.legend([handles[0], handles[1]], [labels[0], labels[1]], loc='upper right', bbox_to_anchor=(1.0, 0.75))
        plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\摧毁目标集决策.svg", dpi=600, format="svg")
        plt.show()

    cluster1 = []
    cluster11 = []
    for i in range(kappa):
        cluster = X[I == i, :]
        if len(cluster) > 0:
            B = max(cluster[:, 3])
        if abs(B - 1) < 1e-4:
            cluster11 = cluster
            cluster1.append(C[i, :])
            break

    cluster2 = []
    for j in range(len(cluster11)):
        for k in range(len(data_Target)):
            if cluster11[j, 0] == data_Target[k, 0]:
                cluster2.append(np.hstack((data2[k, :], data_Target[k, 4])))

    defense = sum(row[6] for row in cluster2) + len(cluster11)
    cluster_1 = [cluster1, cluster2, defense]

    second_list = cluster_1[1]

    filtered_list = [arr for arr in second_list if arr[0] != 117.0]

    cluster_1[1] = filtered_list

    return cluster_1