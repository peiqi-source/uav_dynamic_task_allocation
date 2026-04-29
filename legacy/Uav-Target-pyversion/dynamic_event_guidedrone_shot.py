import numpy as np
import math
import random
import Target_Screen
import joblib
import grouping_ppo
import attack_order_dqn
import plot_figure
from dynamic_event_appear import adjust_allocation
base_location = np.array([0, -16000])

def handle_dynamic_events(data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result, removed_target_labels, new_target_data):

    plot_figure.plot_drones_gui(allocation_result, data_UAV)

    destroyed_guide_uavs_info = []

    num_destroy_guide_uavs = random.randint(1, 2)
    num_destroy_guide_uavs = 2

    if num_destroy_guide_uavs == 1:
        
        selected_cluster = random.choice([3, 6])

        if selected_cluster in allocation_result:
            guide_uav_labels = allocation_result[selected_cluster]["guide_uav_labels"]
            if guide_uav_labels:
                destroyed_label = guide_uav_labels[0]
                print(f"无人机集群{selected_cluster}中的标号为{destroyed_label}的导引无人机被摧毁")
                
                destroyed_guide_uavs_info.append((selected_cluster, destroyed_label))

    elif num_destroy_guide_uavs == 2:
        
        selected_clusters = random.choice([(3, 2), (6, 2)])
        selected_clusters = (6, 2)
        for cluster in selected_clusters:
            if cluster in allocation_result:
                guide_uav_labels = allocation_result[cluster]["guide_uav_labels"]
                if guide_uav_labels:
                    destroyed_label = guide_uav_labels[0]
                    print(f"无人机集群{cluster}中的标号为{destroyed_label}的导引无人机被摧毁")
                    
                    destroyed_guide_uavs_info.append((cluster, destroyed_label))

    if destroyed_guide_uavs_info:
        all_destroyed_cluster_nums = [info[0] for info in destroyed_guide_uavs_info]
        all_destroyed_labels = [info[1] for info in destroyed_guide_uavs_info]
        plot_figure.plot_drones_guidis(allocation_result, data_UAV, destroyed_cluster_num=all_destroyed_cluster_nums,
                                    destroyed_label=all_destroyed_labels)

    for cluster_num, label in zip(all_destroyed_cluster_nums, all_destroyed_labels):
        if cluster_num in allocation_result:
            
            current_guide_uav_labels = allocation_result[cluster_num]["guide_uav_labels"]
            
            if label in current_guide_uav_labels:
                current_guide_uav_labels.remove(label)
            
            allocation_result[cluster_num]["guide_uav_num"] = len(current_guide_uav_labels)

    if num_destroy_guide_uavs == 1:

        env = grouping_ppo.TargetGroupingEnv()
        clustered_data = grouping_ppo.group_targets_guide_1(removed_target_labels, destroy_target_array, new_target_data)
        cluster_centers = plot_figure.plot_cluster_data(clustered_data)

        print("无人机集群资源重分配！")

        if destroyed_guide_uavs_info == [(6, 9)]:

            plot_figure.plot_drones_gui_fenpei_color(allocation_result, data_UAV)

            attack_uav_labels_to_redistribute = allocation_result[6].pop('attack_uav_labels', [])
            target_clusters = [3, 4, 2, 5]
            index = 0
            while attack_uav_labels_to_redistribute:
                current_cluster = target_clusters[index % len(target_clusters)]
                print(f"无人机集群 6 内的攻击无人机 {attack_uav_labels_to_redistribute[0]} 支援到无人机集群 {current_cluster}")
                
                if current_cluster in allocation_result:
                    current_attack_uav_labels = allocation_result[current_cluster].get('attack_uav_labels', [])
                    current_attack_uav_labels.append(attack_uav_labels_to_redistribute.pop(0))
                    allocation_result[current_cluster]['attack_uav_num'] = len(current_attack_uav_labels)
                index += 1
            allocation_result[6] = {
                'guide_uav_num': 0,
                'guide_uav_labels': [],
                'attack_uav_num': 0,
                'attack_uav_labels': [],
                'coordinate': (0, 0),  
                'cluster_center': (0, 0),
                'real_time_position': (0, 0)
            }

            plot_figure.plot_drones_gui_fenpei1(allocation_result, data_UAV)
            
            for cluster_num in allocation_result:
                if cluster_num == 6:
                    
                    allocation_result[cluster_num]['cluster_center'] = cluster_centers.get(5, (0, 0))
                elif cluster_num == 7:
                    
                    allocation_result[cluster_num]['cluster_center'] = cluster_centers.get(6, (0, 0))
                elif cluster_num == 8:
                    
                    allocation_result[cluster_num]['cluster_center'] = cluster_centers.get(7, (0, 0))
                else:
                    
                    allocation_result[cluster_num]['cluster_center'] = cluster_centers.get(cluster_num, (0, 0))
        else:
            
            plot_figure.plot_drones_gui_fenpei_color_3(allocation_result, data_UAV)

            
            attack_uav_labels_to_redistribute = allocation_result[3].pop('attack_uav_labels', [])
            target_clusters = [6, 4, 2, 5]
            index = 0
            while attack_uav_labels_to_redistribute:
                current_cluster = target_clusters[index % len(target_clusters)]
                print(f"无人机集群 3 内的攻击无人机 {attack_uav_labels_to_redistribute[0]} 支援到无人机集群 {current_cluster}")
                
                if current_cluster in allocation_result:
                    current_attack_uav_labels = allocation_result[current_cluster].get('attack_uav_labels', [])
                    current_attack_uav_labels.append(attack_uav_labels_to_redistribute.pop(0))
                    allocation_result[current_cluster]['attack_uav_num'] = len(current_attack_uav_labels)
                index += 1
            allocation_result[3] = {
                'guide_uav_num': 0,
                'guide_uav_labels': [],
                'attack_uav_num': 0,
                'attack_uav_labels': [],
                'coordinate': (0, 0),  
                'cluster_center': (0, 0),
                'real_time_position': (0, 0)
            }
            plot_figure.plot_drones_gui_fenpei1(allocation_result, data_UAV)
            
            for cluster_num in allocation_result:
                if cluster_num == 6:
                    
                    allocation_result[cluster_num]['cluster_center'] = cluster_centers.get(3, (0, 0))
                elif cluster_num == 7:
                    
                    allocation_result[cluster_num]['cluster_center'] = cluster_centers.get(6, (0, 0))
                elif cluster_num == 8:
                    
                    allocation_result[cluster_num]['cluster_center'] = cluster_centers.get(7, (0, 0))
                else:
                    
                    allocation_result[cluster_num]['cluster_center'] = cluster_centers.get(cluster_num, (0, 0))

        for cluster_num, cluster_data in clustered_data.items():
            attack_order_dqn.attack_order_dec(cluster_data, base_location)

    else:

        env = grouping_ppo.TargetGroupingEnv()
        clustered_data = grouping_ppo.group_targets_guidedouble(removed_target_labels, destroy_target_array,
                                                            new_target_data)
        cluster_centers = plot_figure.plot_cluster_data(clustered_data)

        if selected_clusters == (3, 2):
            plot_figure.plot_drones_gui2_fenpei_color_1(allocation_result, data_UAV)
        else:
            plot_figure.plot_drones_gui2_fenpei_color_2(allocation_result, data_UAV)
        
        attack_uav_labels_to_redistribute = allocation_result[2].pop('attack_uav_labels', [])
        target_clusters = [8, 4, 1]
        index = 0
        while attack_uav_labels_to_redistribute:
            current_cluster = target_clusters[index % len(target_clusters)]
            print(f"无人机集群 2 内的攻击无人机 {attack_uav_labels_to_redistribute[0]} 支援到无人机集群 {current_cluster}")
            
            if current_cluster in allocation_result:
                current_attack_uav_labels = allocation_result[current_cluster].get('attack_uav_labels', [])
                current_attack_uav_labels.append(attack_uav_labels_to_redistribute.pop(0))
                allocation_result[current_cluster]['attack_uav_num'] = len(current_attack_uav_labels)
            index += 1
        allocation_result[2] = {
            'guide_uav_num': 0,
            'guide_uav_labels': [],
            'attack_uav_num': 0,
            'attack_uav_labels': [],
            'coordinate': (0, 0),  
            'cluster_center': (0, 0),
            'real_time_position': (0, 0)
        }
        if selected_clusters == (6, 2):
            attack_uav_labels_to_redistribute = allocation_result[6].pop('attack_uav_labels', [])
            target_clusters = [3, 5, 4]
            index = 0
            while attack_uav_labels_to_redistribute:
                current_cluster = target_clusters[index % len(target_clusters)]
                print(f"无人机集群 6 内的攻击无人机 {attack_uav_labels_to_redistribute[0]} 支援到无人机集群 {current_cluster}")
                
                if current_cluster in allocation_result:
                    current_attack_uav_labels = allocation_result[current_cluster].get('attack_uav_labels', [])
                    current_attack_uav_labels.append(attack_uav_labels_to_redistribute.pop(0))
                    allocation_result[current_cluster]['attack_uav_num'] = len(current_attack_uav_labels)
                index += 1
            allocation_result[6] = {
                'guide_uav_num': 0,
                'guide_uav_labels': [],
                'attack_uav_num': 0,
                'attack_uav_labels': [],
                'coordinate': (0, 0),  
                'cluster_center': (0, 0),
                'real_time_position': (0, 0)
            }
        else:
            attack_uav_labels_to_redistribute = allocation_result[3].pop('attack_uav_labels', [])
            target_clusters = [6, 5, 4]
            index = 0
            while attack_uav_labels_to_redistribute:
                current_cluster = target_clusters[index % len(target_clusters)]
                print(f"无人机集群 3 内的攻击无人机 {attack_uav_labels_to_redistribute[0]} 支援到无人机集群 {current_cluster}")
                
                if current_cluster in allocation_result:
                    current_attack_uav_labels = allocation_result[current_cluster].get('attack_uav_labels', [])
                    current_attack_uav_labels.append(attack_uav_labels_to_redistribute.pop(0))
                    allocation_result[current_cluster]['attack_uav_num'] = len(current_attack_uav_labels)
                index += 1
            allocation_result[3] = {
                'guide_uav_num': 0,
                'guide_uav_labels': [],
                'attack_uav_num': 0,
                'attack_uav_labels': [],
                'coordinate': (0, 0),  
                'cluster_center': (0, 0),
                'real_time_position': (0, 0)
            }
        
        plot_figure.plot_drones_gui_fenpei1(allocation_result, data_UAV)

        index = 1
        for cluster_num in allocation_result:
            if allocation_result[cluster_num]['cluster_center'] != (0, 0):
                allocation_result[cluster_num]['cluster_center'] = cluster_centers.get(index,
                                                                                       allocation_result[cluster_num][
                                                                                           'cluster_center'])
                index += 1

        for cluster_num, cluster_data in clustered_data.items():
            attack_order_dqn.attack_order_dec(cluster_data, base_location)

    print(data_Target)

    return destroy_target_array, cluster_centers, clustered_data, allocation_result


