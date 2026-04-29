import numpy as np
import math
import random
import Target_Screen
import joblib
import grouping_ppo
import attack_order_dqn
import plot_figure
import matplotlib.pyplot as plt
from dynamic_event_appear import adjust_allocation


def handle_dynamic_events(data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result):
    data2 = Target_Screen.Target_Screen(data_Target)
    
    cluster_defense = {}
    for cluster_num, target_info_list in clustered_data.items():
        defense_sum = 0
        for target_info in target_info_list:
            defense_sum += target_info[4]  
        cluster_defense[cluster_num] = defense_sum

    
    cluster_attack = {}
    for cluster_num, cluster_info in allocation_result.items():
        attack_sum = 0
        attack_uav_labels = cluster_info["attack_uav_labels"]
        for label in attack_uav_labels:
            drone_rows = [row for row in data_UAV if int(row[0]) == label]
            if drone_rows:
                attack_sum += drone_rows[0][5]
        cluster_attack[cluster_num] = attack_sum


    plot_figure.plot_drones(allocation_result, data_UAV)
    
    grouped_clusters = {
        1: [2, 8],
        2: [1, 3, 4, 6, 7]
    }
    
    selected_drone_clusters = []
    
    for group_num, cluster_list in grouped_clusters.items():
        for cluster in cluster_list:
            
            drone_attack = cluster_attack.get(cluster, 0)
            
            target_defense = cluster_defense.get(cluster, 0)
            if drone_attack - target_defense >= 4:
                selected_drone_clusters.append(cluster)

    final_selected_clusters = []
    
    group_has_multiple_selected = False
    for group_num, cluster_list in grouped_clusters.items():
        group_selected_count = 0
        group_selected = []
        for cluster in cluster_list:
            if cluster in selected_drone_clusters:
                group_selected_count += 1
                group_selected.append(cluster)
        if group_selected_count >= 2:
            group_has_multiple_selected = True
            final_selected_clusters.extend(cluster_list)
        elif group_selected_count == 1:
            for cluster in cluster_list:
                if cluster not in group_selected:
                    final_selected_clusters.append(cluster)
    if not group_has_multiple_selected:
        final_selected_clusters = [cluster for group in grouped_clusters.values() for cluster in group if
                                   cluster not in selected_drone_clusters]

    if final_selected_clusters:
        chosen_cluster = random.choice(final_selected_clusters)
    else:
        chosen_cluster = random.randint(1, 8)

    
    chosen_cluster = 1

    
    if chosen_cluster in allocation_result:
        cluster_info = allocation_result[chosen_cluster]
        
        cluster_info["attack_uav_num"] -= 1
        
        attack_uav_labels = cluster_info["attack_uav_labels"]
        if attack_uav_labels:
            
            remove_index = random.randint(0, len(attack_uav_labels) - 1)

            removed_label = attack_uav_labels[-1]
            
            print(f"被攻击摧毁的无人机是集群 { chosen_cluster } 中的编号为 { removed_label } 的攻击无人机！")
            plot_figure.plot_drones_dis(allocation_result, data_UAV, chosen_cluster, removed_label)
            
            del attack_uav_labels[remove_index]
            
            cluster_attack[chosen_cluster] -= 6
        
        allocation_result[chosen_cluster]["attack_uav_labels"] = attack_uav_labels

    for cluster_id in cluster_attack:
        cluster_attack[cluster_id] += 2

    destroyed_cluster_attack = cluster_attack.get(chosen_cluster, 0)
    target_defense = cluster_defense.get(chosen_cluster, 0)
    if destroyed_cluster_attack < target_defense:
        
        available_support_clusters = []
        for other_cluster_id, other_cluster_attack in cluster_attack.items():
            if other_cluster_id != chosen_cluster:
                remaining_attack = other_cluster_attack - cluster_defense.get(other_cluster_id, 0)
                if remaining_attack >= 6:
                    available_support_clusters.append(other_cluster_id)
        if len(available_support_clusters) >= 2:
            
            min_distance = float('inf')
            closest_cluster = None
            destroyed_cluster_center = allocation_result[chosen_cluster]["real_time_position"]
            for support_cluster_id in available_support_clusters:
                support_cluster_center = allocation_result[support_cluster_id]["real_time_position"]
                
                distance = math.sqrt((destroyed_cluster_center[0] - support_cluster_center[0]) ** 2 +
                                     (destroyed_cluster_center[1] - support_cluster_center[1]) ** 2)
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = support_cluster_id
            print(f"选择无人机集群 {closest_cluster} 对资源受损的无人机集群 {chosen_cluster} 进行支援！")

            target_defense = int(target_defense)
            destroyed_cluster_attack = int(destroyed_cluster_attack)
            support_needed = (target_defense - destroyed_cluster_attack) // 6
            if (target_defense - destroyed_cluster_attack) % 6 != 0:
                support_needed += 1

            support_cluster_info = allocation_result[closest_cluster]
            support_attack_uav_labels = support_cluster_info["attack_uav_labels"]
            if len(support_attack_uav_labels) >= support_needed:
                
                for _ in range(support_needed):
                    removed_support_label = support_attack_uav_labels.pop()
                    
                    attack_uav_labels.append(removed_support_label)
                print(f"从无人机集群 {closest_cluster} 调配 {support_needed} 架攻击无人机支援无人机集群 {chosen_cluster} ；")
                
                allocation_result[chosen_cluster]["attack_uav_num"] += support_needed
                allocation_result[closest_cluster]["attack_uav_num"] -= support_needed
                allocation_result[chosen_cluster]["attack_uav_labels"] = attack_uav_labels
                allocation_result[closest_cluster]["attack_uav_labels"] = support_attack_uav_labels
                
                updated_destroyed_cluster_attack = len(attack_uav_labels) * 6 + 2
                if updated_destroyed_cluster_attack >= target_defense:
                    print(f"无人机集群 {chosen_cluster} 在得到支援后资源要求已满足对抗对应目标群要求！")
                else:
                    print(f"{chosen_cluster}集群在得到支援后攻击力仍未达到要求，还需继续支援或调整策略")
            else:
                print(f"{closest_cluster}集群可提供的攻击无人机数量不足，无法满足对{chosen_cluster}集群的支援需求")
        elif len(available_support_clusters) == 1:
            
            support_cluster_id = available_support_clusters[0]
            print(f"无人机集群 {support_cluster_id} 对被摧毁的无人机集群 {chosen_cluster} 进行支援")

            
            target_defense = int(target_defense)
            destroyed_cluster_attack = int(destroyed_cluster_attack)
            support_needed = (target_defense - destroyed_cluster_attack) // 6
            if (target_defense - destroyed_cluster_attack) % 6 != 0:
                support_needed += 1

            support_cluster_info = allocation_result[support_cluster_id]
            support_attack_uav_labels = support_cluster_info["attack_uav_labels"]
            if len(support_attack_uav_labels) >= support_needed:
                
                for _ in range(support_needed):
                    removed_support_label = support_attack_uav_labels.pop()
                    
                    attack_uav_labels.append(removed_support_label)
                print(f"从无人机集群 {support_cluster_id} 调配 {support_needed} 架攻击无人机支援 {chosen_cluster} 集群")
                
                allocation_result[chosen_cluster]["attack_uav_num"] += support_needed
                allocation_result[support_cluster_id]["attack_uav_num"] -= support_needed
                allocation_result[chosen_cluster]["attack_uav_labels"] = attack_uav_labels
                allocation_result[support_cluster_id]["attack_uav_labels"] = support_attack_uav_labels
                
                updated_destroyed_cluster_attack = len(attack_uav_labels) * 6 + 2
                if updated_destroyed_cluster_attack >= target_defense:
                    print(f"无人机集群 {chosen_cluster} 在得到支援后资源要求已满足对抗对应目标群要求！")
                else:
                    print(f"{chosen_cluster}集群在得到支援后攻击力仍未达到要求，还需继续支援或调整策略")
            else:
                print(f"{support_cluster_id}集群可提供的攻击无人机数量不足，无法满足对{chosen_cluster}集群的支援需求")

        else:
            print("没有可用于支援的无人机集群")
    else:
        print("被摧毁的无人机集群作战资源足够，无需支援！")

    
    plot_figure.plot_drones(allocation_result, data_UAV)

    return destroy_target_array, clustered_data, allocation_result
