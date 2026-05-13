"""历史版本中的无人机数值脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import matplotlib.pyplot as plt

def Length1(x1=None, y1=None, x2=None, y2=None):
    """处理Length1 数据相关业务逻辑。

    参数：
        x1: 横坐标1。
        y1: 纵坐标1。
        x2: 横坐标2。
        y2: 纵坐标2。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    Length = np.sqrt((y2 - y1) * (y2 - y1) + (x2 - x1) * (x2 - x1))
    return Length

def min_max_normalization(data):
    """处理最小值最大值normalization相关业务逻辑。

    参数：
        data: 数据。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    min_value = min(data)
    max_value = max(data)
    normalized_data = [(value - min_value) / (max_value - min_value) for value in data]
    return normalized_data

def UAV_Value(data_UAV = None, m = None, n = None ):

    """处理无人机数值相关业务逻辑。

    参数：
        data_UAV: 数据无人机。
        m: m 数据。
        n: n 数据。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    d = 0   # 导引机数量

    t = 0   # 攻击机数量

    z = 0   # 通信机数量

    mid = data_UAV[:, 3].astype(int)
    mid = mid.tolist()
    d = mid.count(1)
    t = mid.count(2)
    z = mid.count(3)

    # for i in mid:
    #     if i == 1:
    #         d = d + 1
    #         dao[d,:] = data_UAV(i,:)
    #     else:
    #         if data_UAV(i,4) == 2:
    #             t = t + 1
    #             tong[t,:] = data_UAV(i,:)
    #         else:
    #             if data_UAV(i,4) == 3:
    #                 z = z + 1
    #                 gong[z,:] = data_UAV(i,:)

    matrix = data_UAV
    matrix_1 = []
    matrix_2 = []
    matrix_3 = []

    for row in matrix:
        if row[3] == 1:
            matrix_1.append(row)
        elif row[3] == 2:
            matrix_2.append(row)
        elif row[3] == 3:
            matrix_3.append(row)

    matrix_1 = np.array(matrix_1)
    matrix_2 = np.array(matrix_2)
    matrix_3 = np.array(matrix_3)

    dao = matrix_1
    tong = matrix_2
    gong = matrix_3

    plt.rcParams['font.sans-serif'] = ['Microsoft YaHei']
    colors = np.array(['red','green','blue','black'])
    # plt.figure(22)
    # view(45,-25)
    # hold('on')
    # plt.plot(dao(:,2),dao(:,3),'d','color',np.array([0.5,0.5,0.5]),'MarkerFaceColor',colors[0])
    # for i in np.arange(np.arange(1,len(dao(,,1))+1)):
    #     c = dao(i,1)
    #     text(dao(i,2) + 0.01,dao(i,3),num2str(c),'Visible','on')
    #
    # plt.plot(tong(:,2),tong(:,3),'s','color',np.array([0.5,0.5,0.5]),'MarkerFaceColor',colors[2])
    # for j in np.arange(np.arange(1,len(tong(,,1))+1)):
    #     c = tong(j,1)
    #     text(tong(j,2) + 0.01,tong(j,3),num2str(c),'Visible','on')
    #
    # plt.plot(gong(:,2),gong(:,3),'s','color',np.array([0.5,0.5,0.5]),'MarkerFaceColor',colors[2])
    # for k in np.arange(np.arange(1,len(gong(,,1))+1)):
    #     c = gong(k,1)
    #     text(gong(k,2) + 0.01,gong(k,3),num2str(c),'Visible','on')

    plt.figure(figsize=(8, 6))
    plt.scatter(matrix_1[:, 1], matrix_1[:, 2], color='blue', label='observation')
    plt.scatter(matrix_2[:, 1], matrix_2[:, 2], color='green', label='communiaction')
    plt.scatter(matrix_3[:, 1], matrix_3[:, 2], color='red', label='attack')

    plt.title('无人飞行器')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.legend()
    # set(gca, 'LineWidth',1)
    # set(gca,'xtick',0:2:11)
    # set(gca,'ytick',0:2:11)
    # plt.axis('equal')
    # plt.axis(np.array([0,10,0,10]))
    # box('on')
    # hold('off')
    value = np.zeros((d + 1, m - d + 1))
    value1 = np.zeros((d + 1, t + 1))
    value2 = np.zeros((d + 1, z + 1))

    for j in np.arange(1, d + 1).reshape(-1):
        value[j, 0] = dao[j - 1, 0]
        value1[j, 0] = dao[j - 1, 0]
        value2[j, 0] = dao[j - 1, 0]
        for k in np.arange(1, t + 1).reshape(-1):
            value[0, k] = tong[k - 1, 0]
            value1[0, k] = tong[k - 1, 0]
            Length = Length1(dao[j - 1, 1], dao[j - 1, 2], tong[k - 1, 1], tong[k - 1, 2])
            distance = np.abs(dao[j - 1, 4] - tong[k - 1, 4])
            attack = dao[j - 1, 5] + tong[k - 1, 5]
            value[j, k] = 1 * (0.9 ** Length) + 2 * (0.9 ** distance) + attack
            value1[j, k] = 1 * (0.9 ** Length) + 2 * (0.9 ** distance) + attack
        for l in np.arange(1, z + 1).reshape(-1):
            value[0, k + l] = gong[l - 1, 0]
            value2[0, l] = gong[l - 1, 0]
            Length = Length1(dao[j - 1, 1], dao[j - 1, 2], gong[l - 1, 1], gong[l - 1, 2])
            distance = np.abs(dao[j - 1, 4] - gong[l - 1, 4])
            attack = dao[j - 1, 5] + gong[l - 1, 5]
            value[j, k + l] = 1 * (0.9 ** Length) + 2 * (0.9 ** distance) + attack
            value2[j, l] = 1 * (0.9 ** Length) + 2 * (0.9 ** distance) + attack

    b = np.zeros(m)
    data_UAV = np.insert(data_UAV, n, b, axis=1)
    for i in range(1, len(value)):
        sum_value = sum(value[i][1:])
        data_UAV[i - 1][n] = sum_value
    for j in range(1, len(value[0])):
        sum_column = sum(row[j] for row in value[1:])
        data_UAV[i + j - 1][6] = sum_column

    '''
    test1 = data_UAV
    test1[:,2] = np.transpose(mapminmax(np.transpose(test1(:,2)),0,1))
    test1[:,3] = np.transpose(mapminmax(np.transpose(test1(:,3)),0,1))
    test1[:,7] = np.transpose(mapminmax(np.transpose(test1(:,7)),0,1))
    '''

    def min_max_normalization(data):
        """处理最小值最大值normalization相关业务逻辑。

        参数：
            data: 数据。

        返回：
            函数执行结果；具体类型由调用上下文或下游流程决定。
        """
        min_value = min(data)
        max_value = max(data)
        normalized_data = [(value - min_value) / (max_value - min_value) for value in data]
        return normalized_data

    test1 = np.copy(data_UAV)
    test1[:, 1] = min_max_normalization(test1[:, 1])
    test1[:, 2] = min_max_normalization(test1[:, 2])
    test1[:, 6] = min_max_normalization(test1[:, 6])

    return value, d, data_UAV, test1

