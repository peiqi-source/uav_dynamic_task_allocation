import numpy as np
import random
import pandas as pd

data_Target_new = []

num_type_2 = int(200 * 0.2)
for i in range(200):
    if i < num_type_2:
        type_value = 2
        significance_value = 2
    else:
        type_value = 1
        significance_value = 1

    x = random.randint(-2000, 2000)
    y = random.randint(-1500, 1500)
    defense = random.randint(1, 4)
    data_Target_new.append([i + 1, x, y, type_value, defense, significance_value])

data_Target_new = np.array(data_Target_new)

df = pd.DataFrame(data_Target_new, columns=['序号', '位置x', '位置y', '类型type', '防御力defense', '重要性significance'])
excel_file_path = 'D:/西工大/2024秋/大论文/决策/决策/Uav-Target-static-pyversion/test_data/battlefield_target_data.xlsx'
df.to_excel(excel_file_path, index=False)
print(f"数据已成功存入 {excel_file_path} 文件中。")

import matplotlib.pyplot as plt

type_1_data = data_Target_new[data_Target_new[:, 3] == 1]
type_2_data = data_Target_new[data_Target_new[:, 3] == 2]

plt.scatter(type_1_data[:, 1], type_1_data[:, 2], c='b', label='Type 1')
plt.scatter(type_2_data[:, 1], type_2_data[:, 2], c='r', label='Type 2')
plt.xlabel('Position X')
plt.ylabel('Position Y')
plt.title('Distribution of Battlefield Target Elements')
plt.legend()
plt.show()

