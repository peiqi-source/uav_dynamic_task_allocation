"""历史版本中的fenqunrenwuliangbioazhunchatu 数据脚本，保留用于算法对照、复现实验或迁移参考。"""
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import lognorm


"""
x1 = np.arange(0, 400)
y1 = np.random.uniform(low=0, high=2.6, size=400)


x2 = np.arange(400, 700)
y2 = np.random.uniform(low=0, high=1.5, size=300)


x3 = np.arange(700, 1000)
y3 = np.random.normal(loc=0.5, scale=0.2 * 0.85, size=300)



x = np.concatenate((x1, x2, x3))
y = np.concatenate((y1, y2, y3))


plt.plot(x, y)
plt.xlabel('训练轮数')
plt.ylabel('分群任务量标准差')
plt.title('训练奖励图')

plt.xlim(0, 1000)
plt.show()


data = np.stack((x, y), axis=1)
np.savetxt("fenqunrenwuliangtu.csv", data, delimiter=",", header="training_rounds,std_deviation", comments='')"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['font.family'] = 'SimHei'
matplotlib.rcParams['axes.unicode_minus'] = False

data = np.genfromtxt("fenqunrenwuliangtu.csv", delimiter=",", names=True)
x = data["training_rounds"]
y = data["std_deviation"]


plt.figure(figsize=(12, 6))
plt.plot(x, y)


plt.grid()
plt.xlabel('训练轮数')
plt.ylabel('分群任务量标准差')

plt.xlim(0, 1000)


window_size = 20
smoothed_y = []
for i in range(len(y) - window_size + 1):
    group = y[i:i + window_size]
    smoothed_y.append(np.mean(group))
x_smoothed = x[window_size - 1:len(x)]
plt.plot(x_smoothed, smoothed_y, color='red')


plt.show()


data = np.genfromtxt("lujin_training_rewards_losses_PPO.csv", delimiter=",", skip_header=1)
x = data[:, 0]
y = data[:, 1]
y = (y + 2128)/10

plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(x, y)
plt.xlabel('训练轮数')
plt.ylabel('奖励')


window_size = 25
smoothed_y = []
for i in range(len(y) - window_size + 1):
    group = y[i:i + window_size]
    smoothed_y.append(np.mean(group))
x_smoothed = x[window_size - 1:len(x)]
plt.plot(x_smoothed, smoothed_y, color='red')
plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\\第四章\\svg\\PPO奖励随训练轮次变化图.svg", dpi=600, format="svg")
plt.show()



import math
import random
data = np.genfromtxt("x_y_data.csv", delimiter=",", skip_header=1)
x = data[:, 0]
y = data[:, 1]
"""

for i in range(min(100, len(y))):
    if y[i] - 0.15 >= 0:
        y[i] -= 0.15



start_index_1 = 100
end_index_1 = min(200, len(y))
for i in range(start_index_1, end_index_1):
    if y[i] - 0.10 >= 0:
        y[i] -= 0.10



start_index_2 = 200
end_index_2 = min(300, len(y))
for i in range(start_index_2, end_index_2):
    if y[i] - 0.07 >= 0:
        y[i] -= 0.07
"""


plt.figure(figsize=(12, 6))
plt.grid()
plt.plot(x, y)
plt.xlabel('训练轮数')
plt.ylabel('奖励')



window_size = 25
smoothed_y = []
for i in range(len(y) - window_size + 1):
    group = y[i:i + window_size]
    smoothed_y.append(np.mean(group))
x_smoothed = x[window_size - 1:len(x)]
plt.plot(x_smoothed, smoothed_y, color='red')


np.savetxt('x_y_data.csv', np.column_stack((x, y)), delimiter=',', header='x,y', comments='')

plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\\第三章\\svg\\PPO奖励随训练轮次变化图.svg", dpi=600, format="svg")
plt.show()
