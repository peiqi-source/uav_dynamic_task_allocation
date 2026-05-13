"""历史版本中的目标screencore脚本，保留用于算法对照、复现实验或迁移参考。"""
# 开发时间：2024/5/28 19:31
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import minmax_scale


def Distance(x1, y1, x2, y2):
    """处理Distance 数据相关业务逻辑。

    参数：
        x1: 横坐标1。
        y1: 纵坐标1。
        x2: 横坐标2。
        y2: 纵坐标2。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    return np.sqrt((y2 - y1) ** 2 + (x2 - x1) ** 2)


def Target_ScreenCore(data, R, UAVx, UAVy):
    """处理目标screencore相关业务逻辑。

    参数：
        data: 数据。
        R: R 数据。
        UAVx: UAVx 数据。
        UAVy: UAVy 数据。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    DefendedPost = {'num': [], 'x': [], 'y': [], 'type': [], 'defense': [], 'signifcance': [], 'R': [],
                    'distanceUAV': []}
    HVT = {'num': [], 'x': [], 'y': [], 'type': [], 'defense': [], 'signifcance': [], 'distanceUAV': []}

    for i in range(len(data['x'])):
        if data['type'][i] == 2:
            DefendedPost['num'].append(data['num'][i])
            DefendedPost['x'].append(data['x'][i])
            DefendedPost['y'].append(data['y'][i])
            DefendedPost['type'].append(data['type'][i])
            DefendedPost['defense'].append(data['defense'][i])
            DefendedPost['signifcance'].append(data['significance'][i])
            DefendedPost['R'].append(R)
            DefendedPost['distanceUAV'].append(Distance(UAVx, UAVy, DefendedPost['x'][-1], DefendedPost['y'][-1]))

        elif data['type'][i] == 1:
            HVT['num'].append(data['num'][i])
            HVT['x'].append(data['x'][i])
            HVT['y'].append(data['y'][i])
            HVT['type'].append(data['type'][i])
            HVT['defense'].append(data['defense'][i])
            HVT['signifcance'].append(data['significance'][i])
            HVT['distanceUAV'].append(Distance(UAVx, UAVy, HVT['x'][-1], HVT['y'][-1]))

    DPnum = len(DefendedPost['num'])
    HVTnum = len(HVT['num'])

    for i in range(DPnum):
        for j in range(HVTnum):
            HVT['distanceD'] = Distance(DefendedPost['x'][i], DefendedPost['y'][i], HVT['x'][j], HVT['y'][j])
            if HVT['distanceD'] <= DefendedPost['R'][i]:
                DefendedPost['signifcance'][i] += 0.01 * HVT['signifcance'][j]
                HVT['defense'][j] = DefendedPost['defense'][i]

    data1 = {'num': {}, 'x': {}, 'y': {}, 'type': {}, 'defense': {}, 'signifcance': {}, 'distanceUAV': {}}

    for i in range(DPnum):
        num = DefendedPost['num'][i]
        data1['num'][num] = num
        data1['x'][num] = DefendedPost['x'][i]
        data1['y'][num] = DefendedPost['y'][i]
        data1['type'][num] = DefendedPost['type'][i]
        data1['defense'][num] = DefendedPost['defense'][i]
        data1['signifcance'][num] = DefendedPost['signifcance'][i]
        data1['distanceUAV'][num] = DefendedPost['distanceUAV'][i]

    for j in range(HVTnum):
        num = HVT['num'][j]
        data1['num'][num] = num
        data1['x'][num] = HVT['x'][j]
        data1['y'][num] = HVT['y'][j]
        data1['type'][num] = HVT['type'][j]
        data1['defense'][num] = HVT['defense'][j]
        data1['signifcance'][num] = HVT['signifcance'][j]
        data1['distanceUAV'][num] = HVT['distanceUAV'][j]

    # Convert dictionary data1 to lists
    num_list = list(data1['num'].values())
    x_list = list(data1['x'].values())
    y_list = list(data1['y'].values())
    type_list = list(data1['type'].values())
    defense_list = list(data1['defense'].values())
    signifcance_list = list(data1['signifcance'].values())
    distanceUAV_list = list(data1['distanceUAV'].values())

    # Calculate angle
    angle_list = [np.arctan2(x_list[i], y_list[i]) for i in range(len(x_list))]

    # Calculate distanceUAVmin
    distanceUAVmin = min(distanceUAV_list)

    # Normalize distanceUAV_list
    distanceUAV_list_normalized = [val / distanceUAVmin for val in distanceUAV_list]

    # Min-Max scale distanceUAV_list_normalized
    distanceUAV_list_normalized = minmax_scale(distanceUAV_list_normalized)

    # Calculate reward
    reward_list = []
    for k in range(len(num_list)):
        angle = angle_list[k]
        reward = 0.1 * (0.9 ** defense_list[k]) + 10 * 0.5 * signifcance_list[k] + 80 * 0.4 * (
                    0.9 ** distanceUAV_list_normalized[k]) + abs(angle)
        reward_list.append(reward)

    # Initialize flat dictionary data1
    data1 = {'num': [], 'x': [], 'y': [], 'type': [], 'defense': [], 'signifcance': [], 'distanceUAV': [],
             'angle_list': [], 'reward_list': []}

    # 将 data1 转成非嵌套字典
    # Add DefendedPost data to data1
    for i in range(len(DefendedPost['num'])):
        data1['num'].append(DefendedPost['num'][i])
        data1['x'].append(DefendedPost['x'][i])
        data1['y'].append(DefendedPost['y'][i])
        data1['type'].append(DefendedPost['type'][i])
        data1['defense'].append(DefendedPost['defense'][i])
        data1['signifcance'].append(DefendedPost['signifcance'][i])

    # Add HVT data to data1
    for j in range(len(HVT['num'])):
        data1['num'].append(HVT['num'][j])
        data1['x'].append(HVT['x'][j])
        data1['y'].append(HVT['y'][j])
        data1['type'].append(HVT['type'][j])
        data1['defense'].append(HVT['defense'][j])
        data1['signifcance'].append(HVT['signifcance'][j])
        # data1['distanceUAV'].append(HVT['distanceUAV'][j])

    for i in range(len(angle_list)):
        data1['angle_list'].append(angle_list[i])
        data1['reward_list'].append(reward_list[i])
        data1['distanceUAV'].append(distanceUAV_list_normalized[i])

    """
    plt.figure(1)
    plt.title('Strike target')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(-50, 50)
    plt.ylim(-1, 100)
    plt.plot(0, 0, 'bp', markerfacecolor='r', markersize=15)
    plt.text(0 + 0.01, 0, 'Base')
    plt.plot(HVT['x'], HVT['y'], 'o', color=[0.5, 0.5, 0.5], markerfacecolor='g')
    for i in range(len(HVT['x'])):
        plt.text(HVT['x'][i] + 0.01, HVT['y'][i], str(HVT['num'][i]), visible=True)
    plt.plot(DefendedPost['x'], DefendedPost['y'], '^', color=[0.5, 0.5, 0.5], markerfacecolor='b')
    for j in range(len(DefendedPost['x'])):
        plt.text(DefendedPost['x'][j] + 0.01, DefendedPost['y'][j], str(DefendedPost['num'][j]), visible=True)
    plt.legend(['Base', 'High-value target', 'Defensive Position'], loc='upper right')
    plt.grid(True)
    plt.show()
    """

    return data1



