import numpy as np
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

def plot_targets(data_Target):
    
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
        plt.text(x, y, str(index), fontsize=10, ha='center', va='bottom')

    legend_labels = ["地面装甲", "防御阵地", "雷达监测站", "通信枢纽", "指挥部"]
    for i in range(1, 6):
        plt.scatter([], [], marker=marker_dict[i][0], c=marker_dict[i][1], label=legend_labels[i - 1])
    plt.legend(loc='upper right')
    
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('战场目标一览图')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\战场目标一览图.svg", dpi=600, format="svg")
    plt.show()


def plot_uav_counts(data_UAV):
    
    type_mapping = {1: "通信无人机", 2: "导引无人机", 3: "攻击无人机"}
    type_count = {}
    for data in data_UAV:
        uav_type = int(data[3])
        type_name = type_mapping.get(uav_type, f"未知类型{uav_type}")  
        type_count[type_name] = type_count.get(type_name, 0) + 1

    types = list(type_count.keys())
    counts = list(type_count.values())
    bar_positions = np.arange(len(types))

    plt.figure(figsize=(10, 8))
    bars = plt.bar(bar_positions, counts)
    plt.xlabel('无人机类型')
    plt.ylabel('数量')
    plt.title('作战资源一览图')
    plt.xticks(bar_positions, types)

    for bar, count in zip(bars, counts):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width() / 2, height,
                 str(count),
                 ha='center', va='bottom')
    plt.show()


def plot_Destroy_targets(data_Target):
    
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
        plt.text(x, y, str(index), fontsize=10, ha='center', va='bottom')

    legend_labels = ["地面装甲", "防御阵地", "雷达监测站", "通信枢纽", "指挥部"]
    for i in range(1, 6):
        plt.scatter([], [], marker=marker_dict[i][0], c=marker_dict[i][1], label=legend_labels[i - 1])
    plt.legend(loc='upper right')
    
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('摧毁目标集决策')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\摧毁目标集决策平面展示.svg", dpi=600, format="svg")
    plt.show()


def plot_targets_dis(destroy_target_array, removed_target_labels):
    int_removed_target_labels = list(map(int, removed_target_labels))
    marker_dict = {
            1: ('o', 'r'),  
            2: ('s', 'g'),  
            3: ('^', 'b'),  
            4: ('p', 'm'),  
            5: ('D', 'y')  
        }

    plt.figure(figsize=(10, 8))
    for data in destroy_target_array:
        index = int(data[0])
        x = data[1] * 10
        y = data[2] * 10
        target_type = int(data[6])
        
        if index in int_removed_target_labels:
            marker_style = marker_dict.get(target_type, ('o', 'k'))[0]
            edge_color = 'gray'
            face_color = 'gray'
            linestyle = '--'
            
            plt.scatter(x, y, marker=marker_style, c=face_color, edgecolors=edge_color, linewidths=0.1, linestyle=linestyle)
            plt.text(x, y, str(index), fontsize=10, ha='center', va='bottom')
            circle = plt.Circle((x, y), radius=700, fill=False, edgecolor='gray', linestyle='--', linewidth=1.0)
            plt.gca().add_patch(circle)
        else:
            marker, color = marker_dict.get(target_type, ('o', 'k'))
            plt.scatter(x, y, marker=marker, c=color)
            plt.text(x, y, str(index), fontsize=10, ha='center', va='bottom')

    legend_labels = ["地面装甲", "防御阵地", "雷达监测站", "通信枢纽", "指挥部"]
    for i in range(1, 6):
        plt.scatter([], [], marker=marker_dict[i][0], c=marker_dict[i][1], label=legend_labels[i - 1])
    plt.legend(loc='upper right')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('战场目标一览图——现有目标被其他方式摧毁')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\消失目标.svg", dpi=600, format="svg")
    plt.show()


def plot_cluster_data(clustered_data):

    color_map = {
        1: 'r',
        2: 'g',
        3: 'b',
        4: 'm',
        5: 'y',
        6: 'c',
        7: 'orange',
        8: 'purple'
    }
    marker_dict = {
        1: ('o'),  
        2: ('s'),  
        3: ('^'),  
        4: ('p'),  
        5: ('D')  
    }

    plt.figure(figsize=(10, 8))
    legend_labels = ["地面装甲", "防御阵地", "雷达监测站", "通信枢纽", "指挥部"]
    for i in range(1, 6):
        plt.scatter([], [], marker=marker_dict[i][0], color='gray', label=legend_labels[i - 1])
    plt.legend(loc='upper right')

    cluster_centers = {}
    for cluster_num, target_info_list in clustered_data.items():
        cluster_color = color_map.get(cluster_num, 'k')
        
        positions = target_info_list[:, 1:3].astype(float)
        positions *= 10
        center_x = np.mean(positions[:, 0])
        center_y = np.mean(positions[:, 1])
        cluster_centers[cluster_num] = (center_x, center_y)
        
        plt.plot(center_x, center_y, marker='x', color='black', markersize=10)
        for target_info in target_info_list:
            target_type = int(target_info[6])
            marker_style = marker_dict.get(target_type, ('o'))
            x, y = target_info[1:3].astype(float)
            x *= 10
            y *= 10
            plt.scatter(x, y, marker=marker_style, c=cluster_color)
            
            target_label = int(target_info[0])
            plt.text(x, y, str(target_label), fontsize=8, ha='center', va='bottom', color='black')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('目标区域划分示意图')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\6个分群.svg", dpi=600, format="svg")
    plt.show()
    return cluster_centers


def plot_cluster_data_apper_ppo(clustered_data):

    color_map = {
        1: 'r',
        2: 'g',
        3: 'b',
        4: 'm',
        5: 'y',
        6: 'c',
        7: 'orange',
        8: 'purple'
    }
    marker_dict = {
        1: ('o'),  
        2: ('s'),  
        3: ('^'),  
        4: ('p'),  
        5: ('D')  
    }

    plt.figure(figsize=(10, 8))
    legend_labels = ["地面装甲", "防御阵地", "雷达监测站", "通信枢纽", "指挥部"]
    for i in range(1, 6):
        plt.scatter([], [], marker=marker_dict[i][0], color='gray', label=legend_labels[i - 1])
    plt.legend(loc='upper right')

    cluster_centers = {}
    for cluster_num, target_info_list in clustered_data.items():
        cluster_color = color_map.get(cluster_num, 'k')
        
        positions = target_info_list[:, 1:3].astype(float)
        positions *= 10
        center_x = np.mean(positions[:, 0])
        center_y = np.mean(positions[:, 1])
        cluster_centers[cluster_num] = (center_x, center_y)
        
        plt.plot(center_x, center_y, marker='x', color='black', markersize=10)
        for target_info in target_info_list:
            target_type = int(target_info[6])
            marker_style = marker_dict.get(target_type, ('o'))
            x, y = target_info[1:3].astype(float)
            x *= 10
            y *= 10
            plt.scatter(x, y, marker=marker_style, c=cluster_color)
            
            target_label = int(target_info[0])
            plt.text(x, y, str(target_label), fontsize=8, ha='center', va='bottom', color='black')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('目标区域划分示意图')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\出现新增目标后PPO分群.svg", dpi=600, format="svg")
    plt.show()
    return cluster_centers


def plot_targets_appe(destroy_target_array, new_destroy_targets):
    marker_dict = {
        1: ('o', 'r'),  
        2: ('s', 'g'),  
        3: ('^', 'b'),  
        4: ('p', 'm'),  
        5: ('D', 'y')  
    }
    plt.figure(figsize=(10, 8))
    
    for data in destroy_target_array:
        index = int(data[0])
        x = data[1] * 10
        y = data[2] * 10
        target_type = int(data[6])
        marker, color = marker_dict.get(target_type, ('o', 'k'))  
        plt.scatter(x, y, marker=marker, c=color)
        plt.text(x, y, str(index), fontsize=10, ha='center', va='bottom')

    
    for data in new_destroy_targets:
        index = int(data[0])
        x = data[1] * 10
        y = data[2] * 10
        target_type = int(data[6])
        marker, color = marker_dict.get(target_type, ('o', 'k'))  
        plt.scatter(x, y, marker=marker, c=color)
        plt.text(x, y, str(index), fontsize=10, ha='center', va='bottom')
        circle = plt.Circle((x, y), radius=700, fill=False, edgecolor='r', linestyle='--', linewidth=1.0)
        plt.gca().add_patch(circle)

    legend_labels = ["地面装甲", "防御阵地", "雷达监测站", "通信枢纽", "指挥部"]
    for i in range(1, 6):
        plt.scatter([], [], marker=marker_dict[i][0], c=marker_dict[i][1], label=legend_labels[i - 1])
    plt.legend(loc='upper right')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('战场目标一览图——出现新目标后摧毁目标集决策结果')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\出现新目标且RF决策结果.svg", dpi=600, format="svg")
    plt.show()



def plot_targets_appe_no_judge(destroy_target_array, new_destroy_targets):
    marker_dict = {
        1: ('o', 'r'),  
        2: ('s', 'g'),  
        3: ('^', 'b'),  
        4: ('p', 'm'),  
        5: ('D', 'y')  
    }
    plt.figure(figsize=(10, 8))

    
    for data in destroy_target_array:
        index = int(data[0])
        x = data[1] * 10
        y = data[2] * 10
        target_type = int(data[6])
        marker, color = marker_dict.get(target_type, ('o', 'k'))  
        plt.scatter(x, y, marker=marker, c=color)
        plt.text(x, y, str(index), fontsize=10, ha='center', va='bottom')

    
    for data in new_destroy_targets:
        index = int(data[0])
        x = data[1] * 10
        y = data[2] * 10
        target_type = int(data[6])
        marker, color = marker_dict.get(target_type, ('o', 'k'))  
        plt.scatter(x, y, marker=marker, c=color)
        plt.text(x, y, str(index), fontsize=10, ha='center', va='bottom')
        circle = plt.Circle((x, y), radius=700, fill=False, edgecolor='b', linestyle='--', linewidth=1.0)
        plt.gca().add_patch(circle)

    legend_labels = ["地面装甲", "防御阵地", "雷达监测站", "通信枢纽", "指挥部"]
    for i in range(1, 6):
        plt.scatter([], [], marker=marker_dict[i][0], c=marker_dict[i][1], label=legend_labels[i - 1])
    plt.legend(loc='upper right')
    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('战场目标一览图——出现新目标')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\出现新目标.svg", dpi=600, format="svg")
    plt.show()


def plot_drones(allocation_result, data_UAV):
    marker_dict = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    color_list = ["r", "g", "b", "m", "y", "c", "orange", "purple"]  
    plt.figure(figsize=(10, 8))

    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list[cluster_index % 8]

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.5
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict["attack"]
                plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')

        """
        angles = [30, 60, 90, 120, 150, 180, 210, 240, 270, 300, 330, 360]
        for angle in angles:
            angle_rad = np.deg2rad(angle)
            offset_x = 1000 * np.cos(angle_rad)
            offset_y = 1000 * np.sin(angle_rad) * 0.06
            x = center_x + offset_x
            y = center_y + offset_y
            
            plt.scatter(x, y, marker='.', c=cluster_color, alpha=0.5)
            
            circle_x = []
            circle_y = []
            for angle in angles:
                angle_rad = np.deg2rad(angle)
                offset_x = 1000 * np.cos(angle_rad)
                offset_y = 1000 * np.sin(angle_rad) * 0.06
                x = center_x + offset_x
                y = center_y + offset_y
                circle_x.append(x)
                circle_y.append(y)
            circle_x.append(circle_x[0])  
            circle_y.append(circle_y[0])
            plt.plot(circle_x, circle_y, linestyle='dashed', color=cluster_color)

            
            plt.text(center_x + 1000 * np.cos(np.deg2rad(30)), center_y + 1000 * np.sin(np.deg2rad(30)),
                     str(cluster_num), fontsize=8, ha='center', va='bottom', color=cluster_color)
        """

    
    legend_labels = ["攻击无人机", "导引无人机"]
    legend_markers = [marker_dict["attack"][0], marker_dict["guide"][0]]
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c='gray', label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('无人机集群配置示意图')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\攻击机被毁后支援.svg", dpi=600, format="svg")
    plt.show()


def plot_drones_dis(allocation_result, data_UAV, destroyed_cluster_num=None, destroyed_label=None):
    marker_dict = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    color_list = ["r", "g", "b", "m", "y", "c", "orange", "purple"]  
    plt.figure(figsize=(10, 8))

    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list[cluster_index % 8]

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.5
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict["attack"]
                plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')

                
                if cluster_num == destroyed_cluster_num and current_label == destroyed_label:
                    plt.scatter(x, y, marker=marker, c='gray')
                    rect = plt.Rectangle((x - 400, y - 200), 800, 400, fill=False, color='gray', linestyle='dashed')
                    plt.gca().add_patch(rect)


    
    legend_labels = ["攻击无人机", "导引无人机", "被摧毁攻击无人机"]
    legend_markers = [marker_dict["attack"][0], marker_dict["guide"][0], marker_dict["attack"][0]]
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c='gray', label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('无人机集群资源被摧毁示意图')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\攻击无人机摧毁.svg", dpi=600, format="svg")
    plt.show()


def plot_drones_aid(allocation_result, data_UAV, transfer_info):

    if not transfer_info:
        return

    marker_dict = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    color_list = ["r", "g", "b", "m", "y", "c", "orange", "purple"]  
    plt.figure(figsize=(10, 8))

    
    attack_uav_positions = {}
    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list[cluster_index % 8]

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.06
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict["attack"]
                plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')
                
                if cluster_num not in attack_uav_positions:
                    attack_uav_positions[cluster_num] = {}
                attack_uav_positions[cluster_num][current_label] = (x, y)

    
    for info in transfer_info:
        from_cluster, drone_label, to_cluster = info
        from_cluster_color = color_list[(from_cluster - 1) % 8]  
        
        if to_cluster in attack_uav_positions and drone_label in attack_uav_positions[to_cluster]:
            x, y = attack_uav_positions[to_cluster][drone_label]
            plt.scatter(x, y, marker=marker, c=from_cluster_color)
            
            plt.text(x, y, f"from {from_cluster}", fontsize=8, ha='center', va='bottom',
                     color=from_cluster_color)

    
    legend_labels = ["攻击无人机", "导引无人机"]
    legend_markers = [marker_dict["attack"][0], marker_dict["guide"][0]]
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c='gray', label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('无人机集群配置示意图')
    plt.show()


def plot_drones_and_cluster_data(allocation_result, data_UAV, clustered_data):
    """
    在同一张图中绘制无人机集群和目标簇的配置示意图。

    参数:
    allocation_result (dict): 包含各无人机集群详细信息的字典，其中包含"real_time_position"等信息。
    data_UAV (numpy.ndarray): 包含无人机资源数据的二维数组，用于获取无人机类型等相关信息。
    clustered_data (dict): 包含目标簇相关信息的字典，其中键为目标群的群号，值为该目标群内所有目标的属性信息列表。
    """
    
    marker_dict_drones = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    
    color_list_drones = ["r", "g", "b", "m", "y", "c", "orange", "purple"]

    
    color_map_targets = {
        1: 'r',
        2: 'g',
        3: 'b',
        4: 'm',
        5: 'y',
        6: 'c',
        7: 'orange',
        8: 'purple'
    }
    
    marker_dict_targets = {
        1: ('o'),  
        2: ('s'),  
        3: ('^'),  
        4: ('p'),  
        5: ('D')  
    }

    plt.figure(figsize=(10, 8))

    
    legend_labels_targets = ["地面装甲", "防御阵地", "雷达监测站", "通信枢纽", "指挥部"]
    for i in range(1, 6):
        plt.scatter([], [], marker=marker_dict_targets[i][0], color='gray', label=legend_labels_targets[i - 1])
    plt.legend(loc='upper right')

    cluster_centers = {}
    for cluster_num, target_info_list in clustered_data.items():
        cluster_color = color_map_targets.get(cluster_num, 'k')
        
        positions = target_info_list[:, 1:3].astype(float)
        
        center_x = np.mean(positions[:, 0])
        center_y = np.mean(positions[:, 1])
        cluster_centers[cluster_num] = (center_x, center_y)
        
        plt.plot(center_x, center_y, marker='x', color='black', markersize=10)
        for target_info in target_info_list:
            target_type = int(target_info[6])
            marker_style = marker_dict_targets.get(target_type, ('o'))
            x, y = target_info[1:3].astype(float)
            
            
            plt.scatter(x, y, marker=marker_style, c=cluster_color)
            
            target_label = int(target_info[0])
            plt.text(x, y, str(target_label), fontsize=8, ha='center', va='bottom', color='black')

    
    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list_drones[cluster_index % 8]

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict_drones["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.06
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict_drones["attack"]
                plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')

    
    legend_labels = ["地面装甲", "防御阵地", "雷达监测站", "通信枢纽", "指挥部", "攻击无人机", "导引无人机"]
    legend_markers = [marker_dict_targets[i][0] for i in range(1, 6)] + [marker_dict_drones["attack"][0],
                                                                       marker_dict_drones["guide"][0]]
    legend_colors = ['gray'] * 5 + ['gray', 'gray']
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c=legend_colors[i], label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('目标簇与无人机集群配置示意图')
    plt.show()


def plot_drones_guidis(allocation_result, data_UAV, destroyed_cluster_num=None, destroyed_label=None):
    marker_dict = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    color_list = ["r", "g", "b", "m", "y", "c", "orange", "purple"]  
    plt.figure(figsize=(10, 8))

    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list[cluster_index % 8]

        
        """
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                if cluster_num == destroyed_cluster_num and label == destroyed_label:
                    
                    plt.scatter(center_x, center_y, marker=marker, c='gray')
                    
                    
                    
                    rect = plt.Rectangle((center_x - 400, center_y - 20), 800, 40, fill=False, color='gray', linestyle='dashed')
                    plt.gca().add_patch(rect)
                else:
                    plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')
        """

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

                
                if cluster_num in destroyed_cluster_num and label in destroyed_label:
                    plt.scatter(center_x, center_y, marker=marker, c='gray')
                    rect = plt.Rectangle((center_x - 400, center_y - 200), 800, 400, fill=False, color='gray', linestyle='dashed')
                    plt.gca().add_patch(rect)

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.7
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict["attack"]
                plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')

                """
                
                if cluster_num == destroyed_cluster_num and current_label == destroyed_label:
                    plt.scatter(x, y, marker=marker, c='gray')
                    rect = plt.Rectangle((x - 400, y - 20), 800, 40, fill=False, color='gray', linestyle='dashed')
                    plt.gca().add_patch(rect)
                """

    
    legend_labels = ["攻击无人机", "导引无人机", "被摧毁攻击无人机", "被摧毁导引无人机"]
    legend_markers = [marker_dict["attack"][0], marker_dict["guide"][0], marker_dict["attack"][0], 'x']
    legend_colors = ['gray', 'gray', 'gray', 'gray']
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c=legend_colors[i], label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('无人机集群资源被摧毁示意图')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\导引机被毁.svg", dpi=600, format="svg")
    plt.show()


def plot_drones_gui(allocation_result, data_UAV):
    marker_dict = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    color_list = ["r", "g", "b", "m", "y", "c", "orange", "purple"]  
    plt.figure(figsize=(10, 8))

    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list[cluster_index % 8]

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.7
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict["attack"]
                plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')

    
    legend_labels = ["攻击无人机", "导引无人机"]
    legend_markers = [marker_dict["attack"][0], marker_dict["guide"][0]]
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c='gray', label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('无人机集群配置示意图')
    plt.show()


def plot_drones_gui_fenpei1(allocation_result, data_UAV):
    marker_dict = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    color_list = ["r", "g", "b", "m", "y", "c", "orange", "purple"]  
    plt.figure(figsize=(10, 8))

    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list[cluster_index % 8]

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.7
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict["attack"]
                plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')

    
    legend_labels = ["攻击无人机", "导引无人机"]
    legend_markers = [marker_dict["attack"][0], marker_dict["guide"][0]]
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c='gray', label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('无人机集群配置示意图')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\导引机被毁后分配结果.svg", dpi=600, format="svg")
    plt.show()


def plot_drones_gui_fenpei_color(allocation_result, data_UAV):
    marker_dict = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    color_list = ["r", "g", "b", "m", "y", "c", "orange", "purple"]  
    plt.figure(figsize=(10, 8))

    
    color_index_6 = 0
    target_clusters = [3, 4, 2, 5, 6]

    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list[cluster_index % 8]

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.55
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict["attack"]
                
                if cluster_num == 6:
                    target_cluster_index = target_clusters[color_index_6 % len(target_clusters)]
                    target_cluster_color = color_list[(target_cluster_index % 8) - 1]
                    plt.scatter(x, y, marker=marker, c=target_cluster_color)
                    color_index_6 += 1
                else:
                    plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')

    
    legend_labels = ["攻击无人机", "导引无人机"]
    legend_markers = [marker_dict["attack"][0], marker_dict["guide"][0]]
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c='gray', label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('无人机集群配置示意图')
    plt.show()


def plot_drones_gui_fenpei_color_3(allocation_result, data_UAV):
    marker_dict = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    color_list = ["r", "g", "b", "m", "y", "c", "orange", "purple"]  
    plt.figure(figsize=(10, 8))

    
    color_index_6 = 0
    target_clusters = [6, 4, 2, 5]

    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list[cluster_index % 8]

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.55
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict["attack"]
                
                if cluster_num == 3:
                    target_cluster_index = target_clusters[color_index_6 % len(target_clusters)]
                    target_cluster_color = color_list[(target_cluster_index % 8) - 1]
                    plt.scatter(x, y, marker=marker, c=target_cluster_color)
                    color_index_6 += 1
                else:
                    plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')

    
    legend_labels = ["攻击无人机", "导引无人机"]
    legend_markers = [marker_dict["attack"][0], marker_dict["guide"][0]]
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c='gray', label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('无人机集群配置示意图')
    plt.show()


def plot_drones_gui2_fenpei_color_1(allocation_result, data_UAV):
    marker_dict = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    color_list = ["r", "g", "b", "m", "y", "c", "orange", "purple"]  
    plt.figure(figsize=(10, 8))

    
    color_index_6 = 0
    target_clusters = [6, 5, 4]

    color_index_2 = 0
    target_clusters_2 = [8, 4, 1]

    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list[cluster_index % 8]

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.55
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict["attack"]
                
                if cluster_num == 3:
                    target_cluster_index = target_clusters[color_index_6 % len(target_clusters)]
                    target_cluster_color = color_list[(target_cluster_index % 8) - 1]
                    plt.scatter(x, y, marker=marker, c=target_cluster_color)
                    color_index_6 += 1
                elif cluster_num == 2:
                    target_cluster_index_2 = target_clusters_2[color_index_2 % len(target_clusters_2)]
                    target_cluster_color_2 = color_list[(target_cluster_index_2 % 8) - 1]
                    plt.scatter(x, y, marker=marker, c=target_cluster_color_2)
                    color_index_2 += 1
                else:
                    plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')

    
    legend_labels = ["攻击无人机", "导引无人机"]
    legend_markers = [marker_dict["attack"][0], marker_dict["guide"][0]]
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c='gray', label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('无人机集群配置示意图')
    plt.show()


def plot_drones_gui2_fenpei_color_2(allocation_result, data_UAV):
    marker_dict = {
        "attack": ("o", ""),  
        "guide": ("s", "")  
    }
    color_list = ["r", "g", "b", "m", "y", "c", "orange", "purple"]  
    plt.figure(figsize=(10, 8))

    
    color_index_6 = 0
    target_clusters = [3, 5, 4]

    color_index_2 = 0
    target_clusters_2 = [8, 4, 1]

    for cluster_index, (cluster_num, cluster_info) in enumerate(allocation_result.items()):
        
        center_x, center_y = cluster_info["real_time_position"]
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        guide_uav_labels = cluster_info["guide_uav_labels"]

        
        cluster_color = color_list[cluster_index % 8]

        
        for label in guide_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                marker, _ = marker_dict["guide"]
                plt.scatter(center_x, center_y, marker=marker, c=cluster_color)
                
                plt.text(center_x, center_y, str(label), fontsize=8, ha='center', va='bottom', color='black')

        
        num_attack_drones = len(attack_uav_labels)
        if num_attack_drones > 0:
            angles = np.linspace(0, 2 * np.pi, num_attack_drones, endpoint=False)  
            for i in range(num_attack_drones):
                angle = angles[i]
                offset_x = 800 * np.cos(angle)  
                offset_y = 800 * np.sin(angle) * 0.7
                x = center_x + offset_x
                y = center_y + offset_y
                
                current_label = attack_uav_labels[i]
                marker, _ = marker_dict["attack"]
                
                if cluster_num == 6:
                    target_cluster_index = target_clusters[color_index_6 % len(target_clusters)]
                    target_cluster_color = color_list[(target_cluster_index % 8) - 1]
                    plt.scatter(x, y, marker=marker, c=target_cluster_color)
                    color_index_6 += 1
                elif cluster_num == 2:
                    target_cluster_index_2 = target_clusters_2[color_index_2 % len(target_clusters_2)]
                    target_cluster_color_2 = color_list[(target_cluster_index_2 % 8) - 1]
                    plt.scatter(x, y, marker=marker, c=target_cluster_color_2)
                    color_index_2 += 1
                else:
                    plt.scatter(x, y, marker=marker, c=cluster_color)
                
                plt.text(x, y, str(current_label), fontsize=8, ha='center', va='bottom', color='black')

    
    legend_labels = ["攻击无人机", "导引无人机"]
    legend_markers = [marker_dict["attack"][0], marker_dict["guide"][0]]
    for i in range(len(legend_labels)):
        plt.scatter([], [], marker=legend_markers[i], c='gray', label=legend_labels[i])
    plt.legend(loc='upper right')

    plt.xlabel('X轴')
    plt.ylabel('Y轴')
    plt.title('无人机集群配置示意图')
    plt.savefig("D:\\西工大\\2024秋\\大论文\\图库\\python画图\\svg图片夹\\five\\5_3_2\\导引机被毁新配置示意图.svg", dpi=600, format="svg")
    plt.show()

