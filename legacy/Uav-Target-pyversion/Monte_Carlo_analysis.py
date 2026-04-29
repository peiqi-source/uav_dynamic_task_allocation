import numpy as np
import matplotlib.pyplot as plt
import matplotlib


matplotlib.rcParams['font.family'] = 'SimHei'  
matplotlib.rcParams['axes.unicode_minus'] = False  


num_simulations = 10000  
num_targets_per_group = 20  


firepower_range = (5, 20)  
num_drones_range = (5, 10)  
defense_levels = np.random.randint(80, 150, num_targets_per_group)  
distances = np.random.uniform(300, 2000, num_targets_per_group)  


def simulate_damage_probability(num_drones, firepower, defense_levels, distances):
    scale_factor = 600  
    probabilities = (firepower * num_drones * scale_factor) / (defense_levels * (1 + distances))
    probabilities = np.clip(probabilities, 0, 1)  
    return probabilities.mean()  


overall_probabilities = []

for _ in range(num_simulations):
    
    firepower = np.random.uniform(*firepower_range)
    num_drones = np.random.randint(*num_drones_range)

    
    overall_damage_probability = simulate_damage_probability(num_drones, firepower, defense_levels, distances)
    overall_probabilities.append(overall_damage_probability)


damage_threshold = np.percentile(overall_probabilities, 95)
print("推荐的总体毁伤概率阈值为:", damage_threshold)

plt.hist(overall_probabilities, bins=50, color='skyblue', edgecolor='black', density=True)
plt.axvline(damage_threshold, color='red', linestyle='dashed', linewidth=2, label=f'95% 阈值: {damage_threshold:.2f}')
plt.xlabel("任务完成率")
plt.ylabel("频率")
plt.title("总体可完成任务概率分布及阈值")
plt.legend()
plt.show()

import numpy as np
import matplotlib.pyplot as plt
import matplotlib
from mpl_toolkits.mplot3d import Axes3D

matplotlib.rcParams['font.family'] = 'SimHei'  
matplotlib.rcParams['axes.unicode_minus'] = False  

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.patches import Polygon
import matplotlib.image as mpimg


scout_img = mpimg.imread('D:/西工大/2024秋/大论文/图库/侦察机.jpg')
attacker_img = mpimg.imread('D:/西工大/2024秋/大论文/图库/攻击机.jpeg')


num_groups = 3
num_defense_per_group = 1
num_high_value_per_group = np.random.randint(6, 8, num_groups)  
num_targets = num_groups * (num_defense_per_group + num_high_value_per_group.sum())


group_centers = np.array([
    [-1300, 680],
    [-350, -600],
    [800, 510],
    [-490, 1550]
])

positions = []
target_types = []
defense_levels = []

for center in group_centers:
    
    positions.append(center)
    target_types.append('defense')
    defense_levels.append(np.random.randint(3, 5))
    
    for _ in range(np.random.randint(6, 8)):
        
        offset = np.random.rand(2) * 1400 - 200  
        positions.append(center + offset)
        target_types.append('high_value')
        defense_levels.append(2)  

positions = np.array(positions)
target_types = np.array(target_types)

total_defense = sum(defense_levels)
num_scouts = 3  
num_attackers = 12  
firepower = 39

plt.figure(figsize=(10, 6))
plt.scatter(positions[target_types == 'defense', 0], positions[target_types == 'defense', 1],
            marker='^', color='blue', label='防御阵地', s=100)  
plt.scatter(positions[target_types == 'high_value', 0], positions[target_types == 'high_value', 1],
            marker='o', color='orange', label='高价值目标', s=100)

plt.annotate(f"防御力: {defense_levels[0]}", (positions[0, 0], positions[0, 1]),
             textcoords="offset points", xytext=(0,10), ha='center')

high_value_indices = np.where(target_types == 'high_value')[0]
if len(high_value_indices) > 0:
    plt.annotate(f"防御力: {defense_levels[high_value_indices[0]]}",
                 (positions[high_value_indices[0], 0], positions[high_value_indices[0], 1]),
                 textcoords="offset points", xytext=(0,10), ha='center')

plt.scatter(0, -1900, marker='*', color='red', s=300, label='无人机基地')
plt.annotate("无人机基地", (0, -1900), textcoords="offset points", xytext=(0,10), ha='center')


img_size = 600  
plt.imshow(scout_img, extent=(-600, 150, -1650, -1450), aspect='auto')  
plt.text(-250, -1460, '侦察机×3', color='green', fontsize=12, ha='center')  

plt.imshow(attacker_img, extent=(50, 500, -1650, -1450), aspect='auto')  
plt.text(275, -1460, '攻击机×12', color='purple', fontsize=12, ha='center')  

plt.title("目标位置")
plt.xlabel("X 位置")
plt.ylabel("Y 位置")
plt.legend()
plt.xlim(-2200, 2200)
plt.ylim(-2000, 2000)
plt.grid()
plt.show()

plt.figure(figsize=(8, 5))
bar_width = 0.4
resources = [num_scouts, num_attackers, firepower]
resource_labels = ['侦察机数量', '攻击机数量', '火力强度']
plt.bar(resource_labels, resources,width=bar_width, color=['green', 'orange', 'blue'])
plt.title("作战资源参数")
plt.ylabel("数量/强度")
plt.grid(axis='y')
plt.show()

resources = [3, 12, 39, 33]  
resource_labels = ['侦察机数量', '攻击机数量', '总火力强度', '目标群防御力总和']

bar_width = 0.2
x_positions = np.arange(len(resources))

fig, ax = plt.subplots(figsize=(8, 5))

ax.bar(x_positions, resources, width=bar_width, color='

for i, value in enumerate(resources):
    ax.text(i, value + 0.5, str(value), ha='center', va='bottom', fontsize=12)

ax.set_title("作战资源参数", fontsize=16)
ax.set_ylabel("数量/强度")
ax.set_xticks(x_positions)
ax.set_xticklabels(resource_labels, rotation=45, ha='right')


ax.set_xlim(-0.5, len(resources) - 0.5)
plt.tight_layout()
plt.show()


data_Target = np.loadtxt(open('test_data/draw4_5_1.csv'), delimiter=",", skiprows=1)


scout_img = mpimg.imread('D:/西工大/2024秋/大论文/图库/侦察机.jpg')
attacker_img = mpimg.imread('D:/西工大/2024秋/大论文/图库/攻击机.jpeg')

marker_dict = {
        1: ('o', 'r'),  
        2: ('s', 'g'),  
        3: ('^', 'b'),  
        4: ('p', 'm'),  
        5: ('D', 'y')  
    }
plt.figure(figsize=(10, 8))
for data in data_Target:
    index = int(data[0])
    x = data[1] * 10
    y = data[2] * 10
    target_type = int(data[6])
    marker, color = marker_dict.get(target_type, ('o', 'k'))  
    plt.scatter(x, y, marker=marker, c=color)
    

legend_labels = ["地面装甲", "防御阵地", "雷达监测站", "通信枢纽", "指挥部"]
for i in range(1, 6):
    plt.scatter([], [], marker=marker_dict[i][0], c=marker_dict[i][1], label=legend_labels[i - 1])
plt.legend(loc='upper right')



plt.scatter(0, -15000, marker='*', color='red', s=300, label='无人机基地')
plt.annotate("无人机基地", (0, -15000), textcoords="offset points", xytext=(0,10), ha='center')
"""

img_size = 600  
plt.imshow(scout_img, extent=(-600, 150, -1650, -1450), aspect='auto')  
plt.text(-250, -1460, '侦察机×3', color='green', fontsize=12, ha='center')  

plt.imshow(attacker_img, extent=(50, 500, -1650, -1450), aspect='auto')  
plt.text(275, -1460, '攻击机×12', color='purple', fontsize=12, ha='center')  
"""
plt.xlabel('X轴')
plt.ylabel('Y轴')

plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\第四章\\svg\\作战资源场景配置及目标分布一览图.svg", dpi=600, format="svg")
plt.show()

