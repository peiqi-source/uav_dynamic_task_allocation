"""历史版本中的KMeans 算法无人机脚本，保留用于算法对照、复现实验或迁移参考。"""
## (C) Copyright 2012. All rights reserved. Sotiris L Karavarsamis.
# Contact author at sokar@aiia.csd.auth.gr
#
# This is an implementation of the k-means algorithm straight from the
# pseudocode description based on the book 'Introduction to Information
# Retrieval' by Manning, Schutze, Raghavan.
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import Kmeans_UAVCore

def Kmeans_UAV(test1 = None,d = None,data_UAV = None):
    ## set algorithm parameters
    """处理KMeans 算法无人机相关业务逻辑。

    参数：
        test1: test1 数据。
        d: d 数据。
        data_UAV: 数据无人机。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    TOL = 0.004

    ITER = 300

    kappa = d

    # generate random data
    # X = np.array([test1(:,1),test1(:,2),test1(:,3),test1(:,7)])

    colunmns = [0, 1, 2, 6]
    X = np.array([[row[i] for i in colunmns] for row in test1])
    """
     run k-Means on random data
     输入：x,y,z信息、簇数�迭代�数
     输出：簇心坐标�簇类的编号、迭代�数
    """

    #tic
    #上下为测试程序运行时间
    C, I, iter = Kmeans_UAVCore.Kmeans_UAVCore(X, kappa, ITER, TOL)
    #
    #toc
    # show number of iteration taken by k-means

    # print(np.array(['k-means instance took ',str(iter),' iterations to complete']))

    '''
    for i in np.arange(np.arange(0,len(X[:,1]))):
        for j in np.arange(np.arange(0,len(data_UAV[:,1]))):
            if X(i,1) == data_UAV(j,1):
                X[i,2] = data_UAV(j,2)
                X[i,3] = data_UAV(j,3)
    '''

    length1 = X.shape[0]
    length2 = data_UAV.shape[0]
    for i in range(length1):
        for j in range(length2):
            if X[i, 0] == data_UAV[j, 0]:
                X[i, 1] = data_UAV[j, 1]
                X[i, 2] = data_UAV[j, 2]

    '''
    colors = np.array(['red','green','blue','black'])
    plt.figure(21)
    for i in np.arange(1,kappa+1).reshape(-1):
        plot3(X(I == i,2),X(I == i,3),X(I == i,4),'d','color',np.array([0.5,0.5,0.5]),'MarkerFaceColor',colors[i])
        c = X(I == i,1)
        text(X(I == i,2) + 0.01,X(I == i,3) + 0.01,X(I == i,4),num2str(c),'Visible','on')

    plt.title('无人飞�器组网')
    plt.xlabel('X�')
    plt.ylabel('Y�')
    plt.zlabel('价�')
    plt.legend('无人飞�器组网1','无人飞�器组网2','无人飞�器组网3','location','NorthEast')
    set(gca,'LineWidth',1)
    set(gca,'xtick',np.arange(0,10+2,2))
    set(gca,'ytick',np.arange(0,10+2,2))
    set(gca,'ztick',np.arange(0,1+0.2,0.2))
    # axis equal
    # axis([0 10 0 10 0 1])
    grid('on')
    hold('off')
    '''
    # Dictionary to store cluster results
    cluster_results = {}

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    colors = ['red', 'green', 'blue', 'black']
    fig = plt.figure(21)
    ax = fig.add_subplot(111, projection='3d')
    ax.view_init(-45, 35)

    for i in range(kappa):
        idx = np.where(I == (i + 1))[0]
        ax.plot(X[idx, 1], X[idx, 2], X[idx, 3], 'd', color=[0.5, 0.5, 0.5], markerfacecolor=colors[i])

        # Save cluster results
        cluster_results[i + 1] = X[idx, 0].tolist()

        c = X[idx, 0]
        for j in range(len(c)):
            ax.text(X[idx, 1][j] + 0.01, X[idx, 2][j] + 0.01, X[idx, 3][j], str(c[j]), visible=True)

    ax.set_title('无人飞行器组网')
    ax.set_xlabel('X轴')
    ax.set_ylabel('Y轴')
    ax.set_zlabel('value')
    ax.legend(['UAVnet1', 'UAVnet2', 'UAVnet3'], loc='upper right')
    ax.set_xlim([0, 10])
    ax.set_ylim([0, 10])
    ax.set_zlim([0, 1])
    ax.grid(True)
    plt.show()

    '''
    ## 聚类��
    for i in np.arange(1,kappa+1).reshape(-1):
        cluster[i,1] = C(i,:)
    '''

    # 聚类中心
    cluster = []
    for i in range(kappa):
        cluster.append(C[i, :])
        # plt.scatter(C[i, 0], C[i, 1], C[i, 2], color='black', marker='o')

    return cluster, cluster_results
