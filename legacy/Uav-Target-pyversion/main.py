"""历史版本中的main 数据脚本，保留用于算法对照、复现实验或迁移参考。"""
import time
import math
import numpy as np
import UAV
import matplotlib.pyplot as plt
import Target_Screen
import Kmeans_step1
import Kmeans_step2
import attack_order_dqn
import dynamic_event_appear
import dynamic_event_attdrone_shot
import dynamic_event_guidedrone_shot
import plot_figure
import PSO_classify
import dynamic_event_disapper
import time


# 战场参数
attack_payload_per_uav = 6
base_location = np.array([0, -16000])
t = 0
v = 70
update_interval = 30

# 开始计时
start_time = time.time()

data_Target = np.loadtxt(open('test_data/battlefield_target_data.csv'), delimiter=",", skiprows=1)
data_UAV = np.loadtxt(open('test_data/UAV.csv'), delimiter=",", skiprows=1)
# d, attack, cluster_results, test1 = UAV.UAV(data_UAV)

data2 = Target_Screen.Target_Screen(data_Target)

plot_figure.plot_targets(data_Target)
plot_figure.plot_uav_counts(data_UAV)

print("战场态势初始化完成！")

cluster_1 = Kmeans_step1.Kmeans_step1(data_Target, data2, 2)

destroy_target_set = []
cluster_1_indices = [int(arr[0]) for arr in cluster_1[1]]
for index in cluster_1_indices:
    for row in data_Target:
        if int(row[0]) == index:
            target_row = row
            for element in cluster_1[1]:
                if int(element[0]) == index:
                    cluster_row = element
                    combined_data = list(target_row) + list(cluster_row)
                    destroy_target_set.append(combined_data)

plot_figure.plot_Destroy_targets(np.array(destroy_target_set))

# PSO
destroy_target_array = np.array(destroy_target_set)
extracted_data = destroy_target_array[:, :7]
num_clusters = 8
max_iterations = 10
result_clusters, clustered_data, cluster_centers = PSO_classify.target_clustering_visualization(extracted_data,
                                                                                                num_clusters,
                                                                                                max_iterations)
print(clustered_data)

cluster_defense_sums = {}
cluster_numbers = []
for cluster_num, cluster_data in clustered_data.items():
    attack_order_dqn.attack_order_dec(cluster_data, base_location)
    cluster_defense_sum = cluster_data[:, 4].sum()
    cluster_defense_sums[cluster_num] = cluster_defense_sum
    cluster_numbers.append(cluster_num)


allocation_result = {}
guide_uavs = data_UAV[3:11, 0]
assigned_attack_uavs_count = 0
for cluster_num in cluster_numbers:
    required_attack_uavs = np.ceil(cluster_defense_sums[cluster_num] / attack_payload_per_uav).astype(int)
    start_index = 11 + assigned_attack_uavs_count
    attack_uavs = data_UAV[start_index:start_index + required_attack_uavs, 0]
    assigned_attack_uavs_count += len(attack_uavs)
    if len(attack_uavs) > 0:
        x, y = data_UAV[start_index, 1:3]
    else:
        x, y = 0, -28000

    cluster_center = cluster_centers.get(cluster_num, np.array([0, 0]))
    allocation_result[cluster_num] = {
        "guide_uav_num": 1,
        "guide_uav_labels": [int(guide_uavs[cluster_num - 1])],  # 对应群分配的导引机标号
        "attack_uav_num": len(attack_uavs),
        "attack_uav_labels": [int(label) for label in attack_uavs],
        "coordinate": (x, y),
        "cluster_center": (cluster_center[0], cluster_center[1]),
        "real_time_position": (x, y)
    }
# print(allocation_result)

print("无人机集群准备完毕，任务开始！")
end_time = time.time()
print(f"减法操作的执行时间：{end_time - start_time} 秒")

while t < 2200:
    all_arrived = True
    print(f"仿真运行时间：{t}秒")

    for cluster_num, cluster_info in allocation_result.items():
        start_x, start_y = cluster_info["coordinate"]
        target_x, target_y = cluster_info["cluster_center"]
        distance = math.sqrt((target_x - start_x) ** 2 + (target_y - start_y) ** 2)
        if distance > 0:
            all_arrived = False
            flight_time = distance / v
            if t < flight_time:
                ratio = t / flight_time
                current_x = start_x + (target_x - start_x) * ratio
                current_y = start_y + (target_y - start_y) * ratio
            else:
                current_x = target_x
                current_y = target_y

            remaining_distance = math.sqrt((target_x - current_x) ** 2 + (target_y - current_y) ** 2)
            remaining_time = remaining_distance / v if remaining_distance > 0 else 0
            print(f"无人机集群 {cluster_num} 的位置: ({current_x:.1f}, {current_y:.1f})，"
                  f"剩余飞行距离: {remaining_distance:.1f}米，剩余飞行时间: {remaining_time:.1f}秒")

            allocation_result[cluster_num]["real_time_position"] = (current_x, current_y)
        else:
            current_x = start_x
            current_y = start_y
            print(f"无人机集群 {cluster_num} 的位置: ({current_x:.1f}, {current_y:.1f})，"
                  f"剩余飞行距离: 0.0米，剩余飞行时间: 0.0秒")

            allocation_result[cluster_num]["real_time_position"] = (current_x, current_y)

    if t == 300:
        destroy_target_array, \
        cluster_centers, \
        clustered_data, \
        allocation_result, \
        removed_target_labels, \
        new_target_data = dynamic_event_disapper.handle_dynamic_events(
            t, data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result)

    if t == 600:
        destroy_target_array,\
        cluster_centers, \
        clustered_data, \
        allocation_result = dynamic_event_appear.handle_dynamic_events(
            data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result)

    if t == 2010:
        destroy_target_array, \
        clustered_data, \
        allocation_result = dynamic_event_attdrone_shot.handle_dynamic_events(
            data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result)

    """
    if t == 1920:
        destroy_target_array, \
        cluster_centers, \
        clustered_data, \
        allocation_result = dynamic_event_appear.handle_dynamic_events_2(
            data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result)
    """

    if t == 2100:
        destroy_target_array, \
        cluster_centers, \
        clustered_data, \
        allocation_result = dynamic_event_guidedrone_shot.handle_dynamic_events(
            data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result, removed_target_labels, new_target_data)

    if all_arrived:
        print("所有无人机集群已到达目标位置！")
        break

    t += update_interval
    time.sleep(0.1)



print("--------------------------")

# print(data_Target)

# cluster = New_target.handle_new_target_event(data_Target, data_UAV, attack, cluster_1, cluster)
# handle_new_target_event(data_Target, d, attack, cluster)

# cluster = disappearance.handle_disappeared_target(data_Target, cluster, cluster_1, d, data2)
