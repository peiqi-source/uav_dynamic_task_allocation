
import numpy as np
from sklearn.preprocessing import minmax_scale
import Target_ScreenCore
    
def Target_Screen(data_Target = None,attack = None):

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

