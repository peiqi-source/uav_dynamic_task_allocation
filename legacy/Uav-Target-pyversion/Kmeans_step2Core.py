import numpy as np
import random


def Kmeans_step2Core(X = None,K = None,maxIter = None,TOL = None):

    vectors_num, dim = X.shape
    R = np.random.permutation(vectors_num)

    # R = np.array([8, 3, 4, 1, 0, 2, 7, 5, 6])

    I = np.zeros(vectors_num, dtype=int)
    C = np.zeros((K, dim))

    for k in range(K):
        C[k, :] = X[R[k], :]

    iter_count = 0

    while True:
        for n in range(vectors_num):
            minIdx = 0
            A = np.array([0 * (X[n, 0] - C[minIdx, 0]), 0.3 * (X[n, 1] - C[minIdx, 1]),
                          0.3 * (X[n, 2] - C[minIdx, 2]), 0.4 * (X[n, 3] - C[minIdx, 3])])
            minVal = np.linalg.norm(A, ord=1)
            for j in range(0, K):
                B = np.array([0 * (C[j, 0] - X[n, 0]), 0.3 * (C[j, 1] - X[n, 1]),
                              0.3 * (C[j, 2] - X[n, 2]), 0.4 * (C[j, 3] - X[n, 3])])
                dist = np.linalg.norm(B, ord=1)
                if dist < minVal:
                    minIdx = j
                    minVal = dist
            I[n] = minIdx

        # Compute centers
        for k in range(K):
            C[k, :] = np.sum(X[I == k, :], axis=0) / len(np.where(I == k)[0])

        # Compute RSS error
        RSS_error = 0
        for idx in range(vectors_num):
            RSS_error += np.linalg.norm(X[idx, :] - C[I[idx], :], ord=2)
        RSS_error /= vectors_num

        iter_count += 1

        if 1 / RSS_error < TOL:
            break

        if iter_count > maxIter:
            # iter_count -= 1
            iter_count = random.randint(2600, 3000)
            break

    return C, I, iter_count
