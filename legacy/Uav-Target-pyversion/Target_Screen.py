"""历史版本中的目标screen脚本，保留用于算法对照、复现实验或迁移参考。"""

import numpy as np
from sklearn.preprocessing import minmax_scale
import Target_ScreenCore

def Target_Screen(data_Target = None,attack = None):

    """处理目标screen相关业务逻辑。

    参数：
        data_Target: 数据目标。
        attack: 攻击。

    返回：
        函数执行结果；具体类型由调用上下文或下游流程决定。
    """
    R = 20
    x_0 = 0
    y_0 = 0

    test = data_Target.copy()

    data = {}
    data['num'] = test[:, 0]
    data['x'] = test[:, 1]
    data['y'] = test[:, 2]
    data['type'] = test[:, 3]
    data['defense'] = test[:, 4]
    data['significance'] = test[:, 5]

    data1 = Target_ScreenCore.Target_ScreenCore(data, R, x_0, y_0)

    data2 = {}
    data2['num'] = data1['num']
    data2['x'] = minmax_scale(data1['x'], feature_range=(0, 1)).flatten()
    data2['y'] = minmax_scale(data1['y'], feature_range=(0, 1)).flatten()
    data2['reward'] = minmax_scale(data1['reward_list'], feature_range=(0, 1)).flatten()
    data2['distanceUAV'] = minmax_scale(data1['distanceUAV'], feature_range=(0, 1)).flatten()
    data2['angle'] = minmax_scale(data1['angle_list'], feature_range=(0, 1)).flatten()

    data2 = np.array(list(data2.values())).T

    return data2

