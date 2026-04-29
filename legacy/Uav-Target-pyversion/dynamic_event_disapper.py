import numpy as np
import random
import Target_Screen
import grouping_ppo
import attack_order_dqn
import plot_figure
base_location = np.array([0, -16000])

def adjust_allocation(allocation_result, clustered_data, data_UAV, attack_payload_per_uav=6):

    new_cluster_defense_sums = {}
    for cluster_id, targets in clustered_data.items():
        defense_sum = np.sum(targets[:, 4])
        new_cluster_defense_sums[cluster_id] = defense_sum

    transfer_info = []
    for cluster_num in allocation_result:
        required_attack_uavs = np.ceil(new_cluster_defense_sums[cluster_num] / attack_payload_per_uav).astype(int)
        current_attack_uavs = allocation_result[cluster_num]["attack_uav_num"]
        if required_attack_uavs > current_attack_uavs:
            
            surplus_clusters = []
            for other_cluster_num in allocation_result:
                if other_cluster_num!= cluster_num:
                    other_required_attack_uavs = np.ceil(new_cluster_defense_sums[other_cluster_num] / attack_payload_per_uav).astype(int)
                    other_current_attack_uavs = allocation_result[other_cluster_num]["attack_uav_num"]
                    if other_current_attack_uavs > other_required_attack_uavs:
                        surplus_clusters.append((other_cluster_num, other_current_attack_uavs -
                                                 other_required_attack_uavs))

            if surplus_clusters:
                
                min_distance = float('inf')
                closest_surplus_cluster = None
                target_center = clustered_data[cluster_num][0, 1:3]
                for surplus_cluster in surplus_clusters:
                    surplus_cluster_num = surplus_cluster[0]
                    surplus_cluster_center = clustered_data[surplus_cluster_num][0, 1:3]
                    distance = np.linalg.norm(target_center - surplus_cluster_center)
                    if distance < min_distance:
                        min_distance = distance
                        closest_surplus_cluster = surplus_cluster_num

                num_to_transfer = min(required_attack_uavs - current_attack_uavs, allocation_result[closest_surplus_cluster]["attack_uav_num"])
                allocation_result[cluster_num]["attack_uav_num"] += num_to_transfer
                allocation_result[closest_surplus_cluster]["attack_uav_num"] -= num_to_transfer

                source_attack_uavs = allocation_result[closest_surplus_cluster]["attack_uav_labels"][:num_to_transfer]
                target_attack_uavs = allocation_result[cluster_num]["attack_uav_labels"]
                allocation_result[cluster_num]["attack_uav_labels"] = target_attack_uavs + source_attack_uavs
                allocation_result[closest_surplus_cluster]["attack_uav_labels"] = allocation_result[closest_surplus_cluster]["attack_uav_labels"][num_to_transfer:]

                for i in range(num_to_transfer):
                    transfer_info.append((closest_surplus_cluster, source_attack_uavs[i], cluster_num))

    if transfer_info:
        print("无人机集群资源调整信息：")
        for info in transfer_info:
            source_cluster = info[0]
            num_transferred = info[1]
            target_cluster = info[2]
            source_attack_uavs = allocation_result[source_cluster]["attack_uav_labels"][:num_transferred]
            print(f"无人机集群 {source_cluster} 的攻击机 {num_transferred} 去支援无人机集群 {target_cluster}")

    print("调整后的作战资源分配结果：")
    for cluster_num in sorted(allocation_result.keys()):
        print(f"集群 {cluster_num}：导引机数量 {allocation_result[cluster_num]['guide_uav_num']}，导引机标号 {allocation_result[cluster_num]['guide_uav_labels']}，攻击机数量 {allocation_result[cluster_num]['attack_uav_num']}，攻击机标号 {allocation_result[cluster_num]['attack_uav_labels']}")

    return transfer_info, allocation_result


def handle_dynamic_events(t, data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result):

    resource_sufficient = None
    cluster_centers = {}
    removed_target_labels = []
    new_target_data = []

    if t == 300:
        copied_array = destroy_target_array.copy()
        num_targets_to_remove = random.randint(1, 3)
        num_targets_to_remove = 3
        all_target_indices = list(range(len(destroy_target_array)))
        target_indices_to_remove = random.sample(all_target_indices, num_targets_to_remove)
        target_indices_to_remove = [21,25,39]

        for index in sorted(target_indices_to_remove, reverse=True):
            removed_target_label = destroy_target_array[index, 0]
            removed_target_labels.append(removed_target_label)
            destroy_target_array = np.delete(destroy_target_array, index, axis=0)

        print(f"在t = {t}秒时，有{num_targets_to_remove}个目标消失，消失目标标号为：{removed_target_labels}")
        plot_figure.plot_targets_dis(copied_array, removed_target_labels)

        uav_payload_sum = np.sum(data_UAV[:, 5])
        target_attribute_sum = np.sum(destroy_target_array[:, 4])
        resource_sufficient = uav_payload_sum > target_attribute_sum
        print(f"作战资源是否充足：{resource_sufficient}")

        data2 = Target_Screen.Target_Screen(data_Target)
        
        if resource_sufficient:
            supplement_target_labels = [110, 191, 199]
            
            for _ in range(len(removed_target_labels)):
                target_label = supplement_target_labels.pop(0)
                target_row_from_data_Target = None
                target_row_from_data2 = None

                for row in data_Target:
                    if row[0] == target_label:
                        target_row_from_data_Target = row
                        break

                for row in data2:
                    if row[0] == target_label:
                        target_row_from_data2 = row
                        break

                if target_row_from_data_Target is not None and target_row_from_data2 is not None:
                    additional_data = None
                    for row in data_Target:
                        if row[0] == target_label:
                            additional_data = row[4]
                            break
                    combined_row = np.concatenate((target_row_from_data_Target, target_row_from_data2, [additional_data]))
                    new_target_data.append(combined_row)

            if new_target_data:
                destroy_target_array = np.vstack([destroy_target_array, np.array(new_target_data)])

        env = grouping_ppo.TargetGroupingEnv()
        clustered_data = grouping_ppo.group_targets_dis(removed_target_labels, destroy_target_array, new_target_data)
        cluster_centers = plot_figure.plot_cluster_data(clustered_data)

        for cluster_num in allocation_result:
            if cluster_num in cluster_centers:
                allocation_result[cluster_num]["cluster_center"] = cluster_centers[cluster_num]

        transfer_info, allocation_result = adjust_allocation(allocation_result, clustered_data, data_UAV)

        for cluster_num, cluster_data in clustered_data.items():
            attack_order_dqn.attack_order_dec(cluster_data, base_location)

    return destroy_target_array, cluster_centers, clustered_data, allocation_result, removed_target_labels, new_target_data


def handle_dynamic_events_2(t, data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result):

    resource_sufficient = None
    cluster_centers = {}

    copied_array = destroy_target_array.copy()
    num_targets_to_remove = random.randint(1, 3)
    all_target_indices = list(range(len(destroy_target_array)))
    target_indices_to_remove = random.sample(all_target_indices, num_targets_to_remove)

    removed_target_labels = []

    for index in sorted(target_indices_to_remove, reverse=True):
        removed_target_label = destroy_target_array[index, 0]
        removed_target_labels.append(removed_target_label)
        destroy_target_array = np.delete(destroy_target_array, index, axis=0)

    print(f"在t = {t}秒时，有{num_targets_to_remove}个目标消失，消失目标标号为：{removed_target_labels}")
    plot_figure.plot_targets_dis(copied_array, removed_target_labels)

    """
    
    uav_payload_sum = np.sum(data_UAV[:, 5])
    target_attribute_sum = np.sum(destroy_target_array[:, 4])
    resource_sufficient = uav_payload_sum > target_attribute_sum
    print(f"作战资源是否充足：{resource_sufficient}")

    data2 = Target_Screen.Target_Screen(data_Target)
    
    if resource_sufficient:
        supplement_target_labels = [110, 191, 199]
        new_target_data = []
        for _ in range(len(removed_target_labels)):
            target_label = supplement_target_labels.pop(0)
            target_row_from_data_Target = None
            target_row_from_data2 = None

            for row in data_Target:
                if row[0] == target_label:
                    target_row_from_data_Target = row
                    break

            for row in data2:
                if row[0] == target_label:
                    target_row_from_data2 = row
                    break

            if target_row_from_data_Target is not None and target_row_from_data2 is not None:
                additional_data = None
                for row in data_Target:
                    if row[0] == target_label:
                        additional_data = row[4]
                        break
                combined_row = np.concatenate((target_row_from_data_Target, target_row_from_data2, [additional_data]))
                new_target_data.append(combined_row)

        if new_target_data:
            destroy_target_array = np.vstack([destroy_target_array, np.array(new_target_data)])
            
    """

    
    env = grouping_ppo.TargetGroupingEnv()

    for cluster_id in list(clustered_data.keys()):
        target_matrix = clustered_data[cluster_id]
        mask = np.ones(len(target_matrix), dtype=bool)
        for label in removed_target_labels:
            if label in target_matrix[:, 0]:
                row_index = np.where(target_matrix[:, 0] == label)[0]
                mask[row_index] = False
        clustered_data[cluster_id] = target_matrix[mask]

        
        current_target_matrix = clustered_data[cluster_id]
        if len(current_target_matrix) > 0:
            
            position_columns = current_target_matrix[:, 1:3]
            current_target_matrix[:, 1:3] = position_columns / 10
            
            clustered_data[cluster_id] = current_target_matrix

    
    cluster_centers = plot_figure.plot_cluster_data(clustered_data)

    transfer_info, allocation_result = adjust_allocation(allocation_result, clustered_data, data_UAV)

    for cluster_num, cluster_data in clustered_data.items():
        attack_order_dqn.attack_order_dec(cluster_data, base_location)

    return destroy_target_array, cluster_centers, clustered_data, allocation_result
