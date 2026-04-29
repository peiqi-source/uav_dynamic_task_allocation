import numpy as np
import random
import Target_Screen
import joblib
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

def handle_dynamic_events(data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result):
    data2 = Target_Screen.Target_Screen(data_Target)
    new_data_Target = []
    data2_dict = {int(row[0]): row for row in data2}
    for row in data_Target:
        target_label = int(row[0])
        if target_label in data2_dict:
            combined_row = np.concatenate((row, data2_dict[target_label], [row[4]]))
            new_data_Target.append(combined_row)
    new_data_Target = np.array(new_data_Target)

    print("战场态势发生变化，出现新目标!")

    new_appeared_targets = [66, 166, 117, 97, 148]
    
    num_selected_targets = random.randint(1, 4)

    if num_selected_targets > len(new_appeared_targets):
        num_selected_targets = len(new_appeared_targets)
    selected_target_labels = random.sample(new_appeared_targets, num_selected_targets)
    selected_target_labels = [148, 117]

    matched_rows_array = np.array([])
    for label in selected_target_labels:
        for row in data_Target:
            row_label = int(row[0])
            if row_label == label:
                if matched_rows_array.size == 0:
                    matched_rows_array = np.array([row])
                else:
                    matched_rows_array = np.append(matched_rows_array, [row], axis=0)
    
    start_label = 201
    new_matched_rows_array = []
    for index, row in enumerate(matched_rows_array):
        new_row = row.copy()
        new_row[0] = start_label + index
        new_matched_rows_array.append(new_row)
    new_matched_rows_array = np.array(new_matched_rows_array)
    
    plot_figure.plot_targets_appe_no_judge(destroy_target_array, new_matched_rows_array)

    prediction_data = []
    for label in selected_target_labels:
        target_rows = [row for row in new_data_Target if int(row[0]) == label]
        if target_rows:
            target_row = target_rows[0]
            features = target_row[9:12]
            prediction_data.append(features)
    prediction_data = np.array(prediction_data) if prediction_data else np.empty((0, 3))

    loaded_rf_clf = joblib.load('random_forest_model.joblib')
    predictions = loaded_rf_clf.predict(prediction_data) if prediction_data.size > 0 else []

    new_destroy_targets = []
    for i in range(len(selected_target_labels)):
        if predictions.any() and predictions[i] == 1:
            new_destroy_targets.append(selected_target_labels[i])

    complete_new_destroy_targets = []
    for target_label in new_destroy_targets:
        target_rows = [row for row in new_data_Target if int(row[0]) == target_label]
        if target_rows:
            complete_new_destroy_targets.extend(target_rows)
    new_destroy_targets = np.array(complete_new_destroy_targets) if complete_new_destroy_targets else np.empty(
        (0, new_data_Target.shape[1]))

    if new_destroy_targets.size > 0:
        start_label = 201
        num_new_targets = new_destroy_targets.shape[0]
        for i in range(num_new_targets):
            new_destroy_targets[i][0] = start_label + i

    plot_figure.plot_targets_appe(destroy_target_array, new_destroy_targets)

    """
    if new_destroy_targets.any():
        new_rows = [row for row in new_data_Target if int(row[1]) in new_destroy_targets[:, 1].astype(int)]
        destroy_target_array = np.vstack(
            [destroy_target_array, np.array(new_rows)]) if destroy_target_array.size > 0 else np.array(new_rows)
    """
    destroy_target_array = np.vstack([destroy_target_array, new_destroy_targets])

    if new_destroy_targets.size > 0:
        
        cluster_centers = {}
        for cluster_num, target_info_list in clustered_data.items():
            positions = target_info_list[:, 1:3].astype(float)
            positions /= 10
            center_x = np.mean(positions[:, 0])
            center_y = np.mean(positions[:, 1])
            cluster_centers[cluster_num] = (center_x, center_y)

        for target_info in new_destroy_targets[:, :7]:  
            target_x = target_info[1]
            target_y = target_info[2]
            min_distance = float('inf')
            closest_cluster = None
            for cluster_num, center in cluster_centers.items():
                center_x = center[0]
                center_y = center[1]
                distance = np.sqrt((target_x - center_x) ** 2 + (target_y - center_y) ** 2)  
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_num

            target_info_copy = target_info.copy()
            target_info_copy[1] *= 10
            target_info_copy[2] *= 10

            if closest_cluster in clustered_data:
                clustered_data[closest_cluster] = np.vstack([clustered_data[closest_cluster], target_info_copy])
            else:
                clustered_data[closest_cluster] = np.array([target_info_copy])

    for cluster_num, target_info_list in clustered_data.items():
        if len(target_info_list) > 0:
            target_info_list[:, 1:3] /= 10
            clustered_data[cluster_num] = target_info_list
    
    cluster_centers = plot_figure.plot_cluster_data_apper_ppo(clustered_data)
    for cluster_num in allocation_result:
        if cluster_num in cluster_centers:
            allocation_result[cluster_num]["cluster_center"] = cluster_centers[cluster_num]

    transfer_info, allocation_result = adjust_allocation(allocation_result, clustered_data, data_UAV)

    for cluster_num, cluster_data in clustered_data.items():
        attack_order_dqn.attack_order_dec(cluster_data, base_location)

    return destroy_target_array, cluster_centers, clustered_data, allocation_result

def handle_dynamic_events_2(data_Target, destroy_target_array, clustered_data, data_UAV, allocation_result):
    data2 = Target_Screen.Target_Screen(data_Target)
    new_data_Target = []
    data2_dict = {int(row[0]): row for row in data2}
    for row in data_Target:
        target_label = int(row[0])
        if target_label in data2_dict:
            combined_row = np.concatenate((row, data2_dict[target_label], [row[4]]))
            new_data_Target.append(combined_row)
    new_data_Target = np.array(new_data_Target)

    print("战场态势发生变化，出现新目标!")

    new_appeared_targets = [72, 186, 142]
    
    num_selected_targets = random.randint(1, 3)
    if num_selected_targets > len(new_appeared_targets):
        num_selected_targets = len(new_appeared_targets)
    selected_target_labels = random.sample(new_appeared_targets, num_selected_targets)

    matched_rows_array = np.array([])
    for label in selected_target_labels:
        for row in data_Target:
            row_label = int(row[0])
            if row_label == label:
                if matched_rows_array.size == 0:
                    matched_rows_array = np.array([row])
                else:
                    matched_rows_array = np.append(matched_rows_array, [row], axis=0)
    
    start_label = 301
    new_matched_rows_array = []
    for index, row in enumerate(matched_rows_array):
        new_row = row.copy()
        new_row[0] = start_label + index
        new_matched_rows_array.append(new_row)
    new_matched_rows_array = np.array(new_matched_rows_array)
    
    plot_figure.plot_targets_appe_no_judge(destroy_target_array, new_matched_rows_array)

    prediction_data = []
    for label in selected_target_labels:
        target_rows = [row for row in new_data_Target if int(row[0]) == label]
        if target_rows:
            target_row = target_rows[0]
            features = target_row[9:12]
            prediction_data.append(features)
    prediction_data = np.array(prediction_data) if prediction_data else np.empty((0, 3))

    loaded_rf_clf = joblib.load('random_forest_model.joblib')
    predictions = loaded_rf_clf.predict(prediction_data) if prediction_data.size > 0 else []

    new_destroy_targets = []
    for i in range(len(selected_target_labels)):
        if predictions.any() and predictions[i] == 1:
            new_destroy_targets.append(selected_target_labels[i])

    complete_new_destroy_targets = []
    for target_label in new_destroy_targets:
        target_rows = [row for row in new_data_Target if int(row[0]) == target_label]
        if target_rows:
            complete_new_destroy_targets.extend(target_rows)
    new_destroy_targets = np.array(complete_new_destroy_targets) if complete_new_destroy_targets else np.empty(
        (0, new_data_Target.shape[1]))

    if new_destroy_targets.size > 0:
        start_label = 301
        num_new_targets = new_destroy_targets.shape[0]
        for i in range(num_new_targets):
            new_destroy_targets[i][0] = start_label + i

    plot_figure.plot_targets_appe(destroy_target_array, new_destroy_targets)

    """
    if new_destroy_targets.any():
        new_rows = [row for row in new_data_Target if int(row[1]) in new_destroy_targets[:, 1].astype(int)]
        destroy_target_array = np.vstack(
            [destroy_target_array, np.array(new_rows)]) if destroy_target_array.size > 0 else np.array(new_rows)
    """
    destroy_target_array = np.vstack([destroy_target_array, new_destroy_targets])

    if new_destroy_targets.size > 0:
        
        cluster_centers = {}
        for cluster_num, target_info_list in clustered_data.items():
            positions = target_info_list[:, 1:3].astype(float)
            positions /= 10
            center_x = np.mean(positions[:, 0])
            center_y = np.mean(positions[:, 1])
            cluster_centers[cluster_num] = (center_x, center_y)

        for target_info in new_destroy_targets[:, :7]:  
            target_x = target_info[1]
            target_y = target_info[2]
            min_distance = float('inf')
            closest_cluster = None
            for cluster_num, center in cluster_centers.items():
                center_x = center[0]
                center_y = center[1]
                distance = np.sqrt((target_x - center_x) ** 2 + (target_y - center_y) ** 2)  
                if distance < min_distance:
                    min_distance = distance
                    closest_cluster = cluster_num

            target_info_copy = target_info.copy()
            target_info_copy[1] *= 10
            target_info_copy[2] *= 10

            if closest_cluster in clustered_data:
                clustered_data[closest_cluster] = np.vstack([clustered_data[closest_cluster], target_info_copy])
            else:
                clustered_data[closest_cluster] = np.array([target_info_copy])

    for cluster_num, target_info_list in clustered_data.items():
        if len(target_info_list) > 0:
            target_info_list[:, 1:3] /= 10
            clustered_data[cluster_num] = target_info_list
    
    cluster_centers = plot_figure.plot_cluster_data(clustered_data)
    for cluster_num in allocation_result:
        if cluster_num in cluster_centers:
            allocation_result[cluster_num]["cluster_center"] = cluster_centers[cluster_num]

    transfer_info, allocation_result = adjust_allocation(allocation_result, clustered_data, data_UAV)

    for cluster_num, cluster_data in clustered_data.items():
        attack_order_dqn.attack_order_dec(cluster_data, base_location)

    return destroy_target_array, cluster_centers, clustered_data, allocation_result