"""历史版本中的KMeans 算法无人机core脚本，保留用于算法对照、复现实验或迁移参考。"""
## (C) Copyright 2012. All rights reserved. Sotiris L Karavarsamis.
#  Contact the author at <sokar@aiia.csd.auth.gr>
#
#  This is my implementation on the k-means algorithm straight from the
#  pseudocode description of the very same algorithm on the book
#  'Introduction to Information Retrieval' by Manning, Schutze
#  and Raghavan.

import numpy as np
import random

def Kmeans_UAVCore(X = None,K = None,maxIter = None,TOL = None):
    # number of vectors in X X�向量的个�
    """处理KMeans 算法无人机core相关业务逻辑。

    参数：
        X: 横坐标。
        K: K 数据。
        maxIter: 最大值iter。
        TOL: TOL 数据。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    vectors_num,dim = X.shape
    # compute a random permutation of all input vectors 计算�有输入向量的随机排列
    # R = random.randint(1, vectors_num)
    R = list(range(1, vectors_num + 1))
    random.shuffle(R)
    # construct indicator matrix (each entry corresponds to the cluster 构建指示矩阵(每个元素对应于X�每个点的聚类
# of each point in X)
    I = np.zeros((vectors_num,1))
    # construct centers matrix 构造中心矩正
    C = np.zeros((K,dim))
    # take the first K points in the random permutation as the center sead 将随机排列中的前K�点作为中心点
    for k in range(0, K):
        C[k] = X[R[k] - 1]

    # iteration count �代�算
    iter = 0
    # compute new clustering while the cumulative intracluster error in kept 保持累计类内误差不变，计算新的聚类
    # below the maximum allowed error, or the iterative process has not 低于最大允许误差，或迭代过程没
    # exceeded the maximum number of iterations permitted 超过允�的�大迭代�数

    '''
    while 1:

        # find closest point
        for n in np.arange(1,vectors_num+1).reshape(-1):
            # find closest center to current input point 找到与当前输入点�近的��
            minIdx = 1
            A = np.array([0 * (X(n,1) - C(minIdx,1)),(0.2 * (X(n,2) - C(minIdx,2))),(0.2 * (X(n,3) - C(minIdx,3))),(0.6 * 0.95 ** (X(n,4) - C(minIdx,4)))])
            minVal = norm(A,1)
            for j in np.arange(1,K+1).reshape(-1):
                B = np.array([0 * (C(j,1) - X(n,1)),0.2 * (C(j,2) - X(n,2)),0.2 * (C(j,3) - X(n,3)),0.6 * 0.95 ** (C(j,4) - X(n,4))])
                # B = C(j,:) - X(n,:);
                dist = norm(B,1)
                if dist < minVal:
                    minIdx = j
                    minVal = dist
            # assign point to the closter center 指定点到簇中�
            I[n] = minIdx
        # compute centers
        for k in np.arange(1,K+1).reshape(-1):
            C[k,:] = sum(X(I == k,:))
            C[k,:] = C(k,:) / len(find(I == k))
        # compute RSS error
        RSS_error = 0
        for idx in np.arange(1,vectors_num+1).reshape(-1):
            RSS_error = RSS_error + norm(X(idx,:) - C(I(idx),:),2)
        RSS_error = RSS_error / vectors_num
        # increment iteration 增量��
        iter = iter + 1
        # check stopping criteria �查停止准�
        if 1 / RSS_error < TOL:
            break
        if iter > maxIter:
            iter = iter - 1
            break
    '''

    while 1:

        # find closest point
        for n in np.arange(0, vectors_num).reshape(-1):
            # find closest center to current input point 找到与当前输入点�近的��
            minIdx = 1
            A = np.array([0 * (X[n, 0] - C[minIdx - 1, 0]), (0.2 * (X[n, 1] - C[minIdx - 1, 1])),
                          (0.2 * (X[n, 2] - C[minIdx - 1, 2])), (0.6 * 0.95 ** (X[n, 3] - C[minIdx - 1, 3]))])
            # print('A:', A)
            minVal = np.sum(np.abs(A))  # 求1范数
            for j in np.arange(0, K).reshape(-1):
                B = np.array([0 * (C[j, 0] - X[n, 0]), 0.2 * (C[j, 1] - X[n, 1]), 0.2 * (C[j, 2] - X[n, 2]),
                              0.6 * 0.95 ** (C[j, 3] - X[n, 3])])
                # B = C(j,:) - X(n,:);
                dist = np.sum(np.abs(B))
                if dist < minVal:
                    minIdx = j + 1
                    minVal = dist
                # assign point to the closter center 指定点到簇中心
            I[n] = minIdx

        # reply compute centers
        for k in np.arange(1, K + 1).reshape(-1):
            indices = np.where(I == k)[0]  # 找到值为 k 的索引位置
            C[k - 1, :] = np.sum(X[indices], axis=0)  # 对应行的元素进行累加
            C[k - 1, :] = C[k - 1, :] / len(indices)

        # compute RSS error
        RSS_error = 0
        for idx in range(0, vectors_num):
            mid = X[idx] - C[int(I[idx]) - 1]
            norm2_mid = np.linalg.norm(mid)
            RSS_error = RSS_error + norm2_mid
        RSS_error = RSS_error / vectors_num

        # increment iteration 增量迭代
        iter = iter + 1
        # check stopping criteria 检查停止
        if 1 / RSS_error < TOL:
            break
        if iter > maxIter:
            iter = iter - 1
            break

    # disp(['k-means took ' int2str(iter) ' steps to converge']);
    return C,I,iter