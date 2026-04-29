import numpy as np
import matplotlib.pyplot as plt
import Kmeans_step2
import random
import aco_plot
import Particle_algorithm


def calculate_angle(x, y):
    angle = np.arctan2(y, x)  
    angle_degrees = np.degrees(angle)  
    if angle_degrees < 0:
        angle_degrees += 360  
    return angle_degrees



def assess_combat_resources(data_Target, data_UAV):
    total_resources_required = sum(data_Target[:, -1])
    available_resources = sum(data_UAV[:, -1])
    if total_resources_required <= available_resources:
        return True
    else:
        return False


def handle_disappeared_target(data_Target, cluster, cluster_1, d, data2):
    if len(data_Target) == 0:
        print("Error，无目标数据！")
        return data_Target, cluster

    
    num_targets_to_remove = random.randint(1, 3)
    targets_to_remove = random.sample(range(len(data_Target)), num_targets_to_remove)
    biaohao_targets_to_remove = [target + 1 for target in targets_to_remove]
    print("消失目标标号为:", biaohao_targets_to_remove)
    removed_targets = data_Target[targets_to_remove]

    '''
    
    n_data_Target = np.delete(data_Target, targets_to_remove, axis=0)
    plt.figure(1)
    plt.title('战场目标集')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(-50, 50)
    plt.ylim(-1, 100)
    plt.plot(0, 0, 'bp', markerfacecolor='r', markersize=15)
    plt.text(0 + 0.01, 0, 'Base')

    for target in n_data_Target:
        x = target[1]
        y = target[2]
        plt.plot(x, y, 'o', color=[0.5, 0.5, 0.5], markerfacecolor='g')
        plt.text(x + 0.01, y, str(target[0]), visible=True)
    plt.legend(['Base', 'High-value target', 'Defensive Position'], loc='upper right')
    plt.grid(True)
    plt.show()
    '''

    cluster_1_array = np.array(cluster_1[1])
    
    destroy_targets_ids = cluster_1_array[:, 0].astype(int)
    non_destroy_targets_ids = [int(data_Target[i, 0]) for i in range(len(data_Target)) if int(data_Target[i, 0]) not in destroy_targets_ids]
    
    
    

    
    removed_in_destroy_set = [target for target in removed_targets if int(target[0]) in destroy_targets_ids]

    
    if removed_in_destroy_set:
        print("其中属于摧毁目标集的目标标号为:", [int(array[0]) for array in removed_in_destroy_set])

    if not removed_in_destroy_set:
        
        print("消失目标属于非摧毁目标集！")
        colors = ['red', 'green', 'blue', 'black']
        fig = plt.figure(3)
        ax = fig.add_subplot(111, projection='3d')
        ax.view_init(-45, 35)
        ax.scatter(0, 0, 0, c='r', marker='*', label='飞行器基地')
        ax.text(0 + 0.01, 0, 0, '飞行器基地')
        for i, cluster_group in enumerate(cluster):
            if i >= len(colors):  
                color = colors[i % len(colors)]
            else:
                color = colors[i]

            x = cluster_group[:, 1]
            y = cluster_group[:, 2]
            z = cluster_group[:, 3]
            indices = cluster_group[:, 0]

            ax.scatter(x, y, z, c=color, label=f'Cluster {i + 1}')
            for j in range(len(x)):
                ax.text(x[j], y[j], z[j], f'{int(indices[j])}', color='black')
        ax.set_title('摧毁目标集分类')
        ax.set_xlabel('X轴')
        ax.set_ylabel('Y轴')
        ax.set_zlabel('角度')
        ax.legend(loc='upper right')
        ax.set_xticks(np.arange(-50, 51, 20))
        ax.set_yticks(np.arange(0, 101, 20))
        ax.set_zticks(np.arange(0, 1.1, 0.2))
        ax.grid(True)
        plt.show()

        aco_plot.plot_clusters_and_routes(cluster)
        
        return cluster

    
    num_to_add = len(removed_in_destroy_set)

    
    targets_to_add_ids = random.sample(non_destroy_targets_ids, num_to_add)
    print("补充目标标号为:", targets_to_add_ids)
    targets_to_add = np.array([data_Target[data_Target[:, 0] == target_id][0] for target_id in targets_to_add_ids])
    data2_to_add = np.array([data2[data2[:, 0] == target_id][0] for target_id in targets_to_add_ids])

    
    new_targets = np.hstack((data2_to_add, targets_to_add[:, 4:5]))

    
    cluster_1_array = np.concatenate((cluster_1_array, new_targets))
    cluster_1[1] = cluster_1_array

    
    

    
    targets = cluster_1[1]

    
    targets = np.array([target for target in targets if target[0] not in biaohao_targets_to_remove])
    

    
    for target in targets:
        original_position = data_Target[data_Target[:, 0] == target[0], 1:3]
        x, y = original_position[0]
        angle = calculate_angle(x, y)
        target[-2] = angle

    num_clusters = 3
    best_solution, best_value, best_assignment = Particle_algorithm.pso_optimization(targets, num_clusters)
    print("最佳分配方案:", best_assignment)

    colors = ['red', 'green', 'blue']
    
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')

    
    ax.scatter(0, 0, 0, c='r', marker='*', label='飞行器基地')
    ax.text(0 + 0.01, 0, 0, '飞行器基地')
    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, x in enumerate(best_assignment) if x == cluster_id]
        cluster_targets = targets[cluster_indices]

        
        original_positions = np.array(
            [data_Target[data_Target[:, 0] == target[0], 1:4] for target in cluster_targets]).reshape(-1, 3)
        x = original_positions[:, 0]  
        y = original_positions[:, 1]  
        z = cluster_targets[:, 5]  
        indices = cluster_targets[:, 0]

        ax.scatter(x, y, z, color=colors[cluster_id], label=f'Cluster {cluster_id + 1}')

        for j in range(len(x)):
            ax.text(x[j], y[j], z[j], f'{int(indices[j])}', color='black')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Angle')
    ax.set_title('摧毁目标集分类')
    ax.legend()
    plt.show()

    
    updated_clusters = []
    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, x in enumerate(best_assignment) if x == cluster_id]
        cluster_targets = targets[cluster_indices]

        
        updated_cluster = np.array(
            [[target[0], data_Target[data_Target[:, 0] == target[0], 1][0],
              data_Target[data_Target[:, 0] == target[0], 2][0],
              data2[data2[:, 0] == target[0], -1][0]] for target in cluster_targets]
        )
        updated_cluster = updated_cluster[updated_cluster[:, -1].argsort()[::-1]]
        updated_clusters.append(updated_cluster)

    cluster = updated_clusters
    aco_plot.plot_clusters_and_routes(cluster)

    '''
    num_clusters = 3
    
    
    best_solution, best_value, best_assignment = Particle_algorithm.pso_optimization(targets, num_clusters)
    print("最佳分配方案:", best_assignment)

    colors = ['red', 'green', 'blue']
    
    fig = plt.figure(3)
    ax = fig.add_subplot(111, projection='3d')

    
    ax.scatter(0, 0, 0, c='r', marker='*', label='飞行器基地')
    ax.text(0 + 0.01, 0, 0, '飞行器基地')
    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, x in enumerate(best_assignment) if x == cluster_id]
        cluster_targets = targets[cluster_indices]

        
        
        extracted_positions = [data_Target[data_Target[:, 0] == target[0], 1:4] for target in cluster_targets]
        print("Extracted positions:", extracted_positions)

        
        extracted_positions_array = np.array(extracted_positions)
        print("Extracted positions array shape:", extracted_positions_array.shape)
        

        
        original_positions = []
        for target in cluster_targets:
            matching_rows = data_Target[data_Target[:, 0] == target[0]]
            if len(matching_rows) > 0:
                original_positions.append(matching_rows[0, 1:4])
            else:
                print(f"Warning: Target ID {target[0]} not found in data_Target")
                original_positions.append([np.nan, np.nan, np.nan])

        original_positions = np.array(original_positions)
        x = original_positions[:, 0]  
        y = original_positions[:, 1]  
        z = cluster_targets[:, 5]  
        indices = cluster_targets[:, 0]

        ax.scatter(x, y, z, color=colors[cluster_id], label=f'Cluster {cluster_id + 1}')
        for j in range(len(x)):
            ax.text(x[j], y[j], z[j], f'{int(indices[j])}', color='black')

    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')
    ax.set_zlabel('Angle')
    ax.set_title('摧毁目标集分类')
    ax.legend()
    plt.show()

    
    cluster = []
    for cluster_id in range(num_clusters):
        cluster_indices = [i for i, x in enumerate(best_assignment) if x == cluster_id]
        cluster_targets = targets[cluster_indices]

        
        updated_positions = []
        for target in cluster_targets:
            matching_rows = data_Target[data_Target[:, 0] == target[0]]
            if len(matching_rows) > 0:
                original_position = matching_rows[0, 1:4]
                angle = cluster_targets[-1]
                
                posx = original_position[0]
                posy = original_position[1]
                if np.isfinite(posx) and np.isfinite(posy):
                    updated_positions.append([target[0], posx, posy, angle])
                else:
                    print(f"Warning: Invalid position values for target ID {target[0]} (x: {posx}, y: {posy})")
            else:
                print(f"Warning: Target ID {target[0]} not found in data_Target")

        cluster.append(np.array(updated_positions))

    aco_plot.plot_clusters_and_routes(cluster)
    '''

    return cluster
