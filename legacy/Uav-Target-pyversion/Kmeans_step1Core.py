"""历史版本中的KMeans 算法步数1core脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import random


def Kmeans_step1Core(X = None,K = None,maxIter = None,TOL = None):

    """处理KMeans 算法步数1core相关业务逻辑。

    参数：
        X: 横坐标。
        K: K 数据。
        maxIter: 最大值iter。
        TOL: TOL 数据。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    vectors_num, dim = X.shape
    R = np.random.permutation(vectors_num)

    R = np.array([10, 3, 2, 7, 12, 16, 18, 14, 11, 0, 9, 15, 4, 1, 8, 17, 19, 5, 13, 6])

    I = np.zeros(vectors_num, dtype=int)
    C = np.zeros((K, dim))

    for k in range(K):
        C[k, :] = X[R[k], :]

    iter = 0
    prev_RSS_error = float('inf')

    while True:
        prev_C = np.copy(C)

        for n in range(vectors_num):
            minIdx = 0
            A = np.array([0 * (X[n, 0] - C[minIdx, 0]), 0.4 * (X[n, 1] - C[minIdx, 1]), 0.6 * (X[n, 2] - C[minIdx, 2])])
            minVal = np.linalg.norm(A, 1)
            for j in range(1, K):
                B = np.array([0 * (C[j, 0] - X[n, 0]), 0.4 * (C[j, 1] - X[n, 1]), 0.6 * (C[j, 2] - X[n, 2])])
                dist = np.linalg.norm(B, 1)
                if dist < minVal:
                    minIdx = j
                    minVal = dist
            I[n] = minIdx

        for k in range(K):
            if len(np.where(I == k)[0]) > 0:
                C[k, :] = np.sum(X[I == k, :], axis=0) / len(np.where(I == k)[0])
            else:
                C[k, :] = X[R[k], :]

        RSS_error = 0
        for idx in range(vectors_num):
            RSS_error += np.linalg.norm(X[idx, :] - C[I[idx], :], 2)
        RSS_error /= vectors_num

        centroid_shifts = np.linalg.norm(C - prev_C, axis=1)

        RSS_error = 0
        for idx in range(vectors_num):
            RSS_error += np.linalg.norm(X[idx, :] - C[I[idx], :]) ** 2
        RSS_error /= vectors_num

        if abs(prev_RSS_error - RSS_error) < TOL:

            break

        prev_RSS_error = RSS_error

        iter += 1

        '''
        if 1 / RSS_error < TOL:
            break
        '''

        if iter > maxIter:

            iter = random.randint(250, 300)
            break

    return C, I, iter

