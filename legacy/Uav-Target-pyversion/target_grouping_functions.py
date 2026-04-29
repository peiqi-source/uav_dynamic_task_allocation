import numpy as np


def group_targets_dis(removed_target_labels, destroy_target_array, new_target_data):

    given_clusters_1 = {
        1: [123, 171, 158, 89, 23, 124, 128, 104, 10, 141],
        2: [8, 14, 13, 9, 21, 19, 38],
        3: [33, 40, 147, 174, 196, 3, 136, 138],
        4: [2, 12, 4, 22, 1, 20, 37],
        5: [11, 7, 29, 28, 5, 30, 17],
        6: [6, 34, 36, 39, 106, 68, 67],
        7: [15, 24, 35, 43, 57, 119, 149, 151, 167],
        8: [16, 18, 27, 25, 26, 32, 31]
    }

    
    cluster_indices = {}
    for cluster_id, target_labels in given_clusters_1.items():
        indices = []
        for index, target in enumerate(destroy_target_array):
            if target[0] in target_labels:
                indices.append(index)
        cluster_indices[cluster_id] = np.array(indices)

    
    cluster_centers = {}
    for cluster_id, indices in cluster_indices.items():
        target_positions = destroy_target_array[indices, 1:3]
        center_x = np.mean(target_positions[:, 0])
        center_y = np.mean(target_positions[:, 1])
        cluster_centers[cluster_id] = np.array([center_x, center_y])

    
    for new_target in new_target_data:
        new_target_label = new_target[0]
        new_target_position = new_target[1:3]
        min_distance = float('inf')
        closest_cluster = None
        for cluster_id, center in cluster_centers.items():
            distance = np.linalg.norm(new_target_position - center)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        if closest_cluster is not None:
            given_clusters_1[closest_cluster].append(new_target_label)

    
    for label in removed_target_labels:
        for cluster_id in given_clusters_1:
            if label in given_clusters_1[cluster_id]:
                given_clusters_1[cluster_id].remove(label)

    
    clustered_data = {}
    for cluster_id in given_clusters_1:
        target_info_list = []
        for target_label in given_clusters_1[cluster_id]:
            for target in destroy_target_array:
                if target[0] == target_label:
                    target_info_list.append(target[:7])  
                    break
        clustered_data[cluster_id] = np.array(target_info_list)


    return clustered_data


def group_targets_guide_1(removed_target_labels, destroy_target_array, new_target_data):

    given_clusters_1 = {
        1: [123, 171, 158, 89, 23, 124, 128, 104, 10, 141],     
        2: [8, 14, 13, 9, 21, 19, 38, 1, 2],        
        3: [33, 40, 147, 174, 196, 3, 136, 138, 34,  106],
        4: [12, 4, 22, 20, 37, 36, 39],                 
        5: [11, 7, 29, 28, 5, 30, 17, 6, 67],                  
        6: [15, 24, 35, 43, 57, 119, 149, 151, 167, 68],
        7: [16, 18, 27, 25, 26, 32, 31]
    }

    
    for label in removed_target_labels:
        for cluster_id in given_clusters_1:
            if label in given_clusters_1[cluster_id]:
                given_clusters_1[cluster_id].remove(label)

    
    cluster_indices = {}
    for cluster_id, target_labels in given_clusters_1.items():
        indices = []
        for index, target in enumerate(destroy_target_array):
            if target[0] in target_labels:
                indices.append(index)
        cluster_indices[cluster_id] = np.array(indices)

    
    cluster_centers = {}
    for cluster_id, indices in cluster_indices.items():
        target_positions = destroy_target_array[indices, 1:3]
        center_x = np.mean(target_positions[:, 0])
        center_y = np.mean(target_positions[:, 1])
        cluster_centers[cluster_id] = np.array([center_x, center_y])

    
    for new_target in new_target_data:
        new_target_label = new_target[0]
        new_target_position = new_target[1:3]
        min_distance = float('inf')
        closest_cluster = None
        for cluster_id, center in cluster_centers.items():
            distance = np.linalg.norm(new_target_position - center)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        if closest_cluster is not None:
            given_clusters_1[closest_cluster].append(new_target_label)

    
    new_targets_list = []
    for row in destroy_target_array:
        target_label = int(row[0])
        if target_label >= 201:
            new_targets_list.append(row)

    
    for target in new_targets_list:
        target_label = int(target[0])
        target_position = target[1:3]
        min_distance = float('inf')
        closest_cluster = None
        for cluster_id, center in cluster_centers.items():
            distance = np.linalg.norm(target_position - center)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        if closest_cluster is not None:
            given_clusters_1[closest_cluster].append(target_label)

    
    clustered_data = {}
    for cluster_id in given_clusters_1:
        target_info_list = []
        for target_label in given_clusters_1[cluster_id]:
            for target in destroy_target_array:
                if target[0] == target_label:
                    target_info_list.append(target[:7])  
                    break
        clustered_data[cluster_id] = np.array(target_info_list)

    return clustered_data


def group_targets_guidedouble(removed_target_labels, destroy_target_array, new_target_data):
    given_clusters_1 = {
        1: [123, 171, 158, 89, 23, 124, 128, 104, 10, 141, 19, 38],     
        2: [33, 40, 147, 174, 196, 3, 136, 138, 34,  106],
        3: [12, 4, 22, 20, 37, 39, 9, 21, 1, 2],                 
        4: [11, 7, 29, 28, 5, 30, 17, 6, 67, 36],                  
        5: [15, 24, 35, 43, 57, 119, 149, 151, 167, 68],
        6: [16, 18, 27, 25, 26, 32, 31, 8, 14, 13]  
    }

    
    for label in removed_target_labels:
        for cluster_id in given_clusters_1:
            if label in given_clusters_1[cluster_id]:
                given_clusters_1[cluster_id].remove(label)

    
    cluster_indices = {}
    for cluster_id, target_labels in given_clusters_1.items():
        indices = []
        for index, target in enumerate(destroy_target_array):
            if target[0] in target_labels:
                indices.append(index)
        cluster_indices[cluster_id] = np.array(indices)

    
    cluster_centers = {}
    for cluster_id, indices in cluster_indices.items():
        target_positions = destroy_target_array[indices, 1:3]
        center_x = np.mean(target_positions[:, 0])
        center_y = np.mean(target_positions[:, 1])
        cluster_centers[cluster_id] = np.array([center_x, center_y])

    
    for new_target in new_target_data:
        new_target_label = new_target[0]
        new_target_position = new_target[1:3]
        min_distance = float('inf')
        closest_cluster = None
        for cluster_id, center in cluster_centers.items():
            distance = np.linalg.norm(new_target_position - center)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        if closest_cluster is not None:
            given_clusters_1[closest_cluster].append(new_target_label)

    print(given_clusters_1)
    
    if 199.0 in given_clusters_1[2]:
        given_clusters_1[2].remove(199.0)

    
    given_clusters_1[3].append(199.0)
    new_targets_list = []
    for row in destroy_target_array:
        target_label = int(row[0])
        if target_label >= 201:
            new_targets_list.append(row)

    
    for target in new_targets_list:
        target_label = int(target[0])
        target_position = target[1:3]
        min_distance = float('inf')
        closest_cluster = None
        for cluster_id, center in cluster_centers.items():
            distance = np.linalg.norm(target_position - center)
            if distance < min_distance:
                min_distance = distance
                closest_cluster = cluster_id
        if closest_cluster is not None:
            given_clusters_1[closest_cluster].append(target_label)

    
    clustered_data = {}
    for cluster_id in given_clusters_1:
        target_info_list = []
        for target_label in given_clusters_1[cluster_id]:
            for target in destroy_target_array:
                if target[0] == target_label:
                    target_info_list.append(target[:7])  
                    break
        clustered_data[cluster_id] = np.array(target_info_list)

    return clustered_data

