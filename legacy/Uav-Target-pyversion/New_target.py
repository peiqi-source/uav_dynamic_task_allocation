import random
import numpy as np
import Target_Screen
import matplotlib.pyplot as plt
import joblib
import Particle_algorithm
import aco_plot
import time
from sklearn.ensemble import RandomForestClassifier


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


def handle_new_target_event(data_Target, data_UAV, attack, cluster_1, cluster):
    num_new_targets = random.randint(1, 3)  
    new_targets = []

    for _ in range(num_new_targets):

        new_target_index = len(data_Target) + 1
        new_target_x = random.randint(-50, 50)
        new_target_y = random.randint(0, 100)
        new_target_type = random.randint(1, 2)
        new_target_defense = random.randint(1, 4)
        new_target_importance = random.randint(1, 2)

        new_target_info = [new_target_index, new_target_x, new_target_y, new_target_type, new_target_defense,
                       new_target_importance]
        new_targets.append(new_target_info)
        data_Target = np.vstack([data_Target, new_target_info])

    '''
    plt.figure(1)
    plt.title('Strike target')
    plt.xlabel('X-axis')
    plt.ylabel('Y-axis')
    plt.xlim(-50, 50)
    plt.ylim(-1, 100)
    plt.plot(0, 0, 'bp', markerfacecolor='r', markersize=15)
    plt.text(0 + 0.01, 0, 'Base')
    '''
    
    start_time = time.time()
    data2 = Target_Screen.Target_Screen(data_Target, attack)

    
    rf_clf = joblib.load('random_forest_model.joblib')
    new_features = data2[-num_new_targets:, 3:6]
    predictions = rf_clf.predict(new_features)
    destroy_targets = []

    for i, (prediction, new_target_info) in enumerate(zip(predictions, new_targets),
                                                      start=len(data_Target) - num_new_targets + 1):
        print(f"New target {new_target_info[0]}: {'Destroy' if prediction == 1 else 'Do not destroy'}")
        if prediction == 1:
            new_target_index = new_target_info[0]
            target_data = data2[data2[:, 0] == new_target_index][:, :6]
            if target_data.size > 0:
                target_data = target_data[0]
                target_data = np.append(target_data, new_target_info[4])  
                destroy_targets.append(target_data)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"使用RF预测耗时: {execution_time:.3f} 秒")

    """
    for i, (prediction, new_target_info) in enumerate(zip(predictions, new_targets),
                                                      start=len(data_Target) - num_new_targets + 1):
        print(f"New target {new_target_info[0]}: {'Destroy' if prediction == 1 else 'Do not destroy'}")

    cluster_1_add = Kmeans_step1.Kmeans_step1(data_Target, data2, 2, caller='handle_new_target_event')

    second_array = np.array(cluster_1_add[1])
    destroy_targets_indices = second_array[:, 0]

    added_to_destroy_set = False

    for new_target_info in new_targets:
        new_target_index = new_target_info[0]
        if new_target_index in destroy_targets_indices:
            print(f"New target {new_target_index} added to destroy targets set.")
            
            cluster_1[1] = np.vstack([cluster_1[1], second_array[np.where(second_array[:, 0] == new_target_index)][0]])
            added_to_destroy_set = True
    """

    """
    if new_target_index in destroy_targets_indices:
        print(f"New target {new_target_index} added to destroy targets set.")
        
        cluster_1[1] = np.vstack([cluster_1[1], second_array[-1]])

        
        
        cluster = Kmeans_step2.Kmeans_step2(data_Target, cluster_1, d)
    """

    if destroy_targets:
        destroy_targets = np.array(destroy_targets)
        cluster_1[1] = np.vstack([cluster_1[1], destroy_targets])

        
        if not assess_combat_resources(cluster_1[1], data_UAV):
            print(f"进攻资源有限，筛选摧毁目标集!")
            
            num_to_remove = np.sum(predictions == 1)
            sorted_indices = np.argsort(cluster_1[1][:, 3])  
            removed_indices = sorted_indices[:num_to_remove]
            removed_targets = cluster_1[1][removed_indices, 0]  
            cluster_1[1] = np.delete(cluster_1[1], removed_indices, axis=0)

            for removed_index in removed_targets:
                print(f"受进攻资源限制，目标 {int(removed_index)} 从摧毁目标集清除!")

        start_time = time.time()
        targets = cluster_1[1]
        

        plt.figure(1)
        plt.title('摧毁目标集')
        plt.xlabel('X-axis')
        plt.ylabel('Y-axis')
        plt.xlim(-50, 50)
        plt.ylim(-1, 100)
        plt.plot(0, 0, 'bp', markerfacecolor='r', markersize=15)
        plt.text(0 + 0.01, 0, 'Base')

        for target in cluster_1[1]:
            original_position = data_Target[data_Target[:, 0] == target[0], 1:3]
            original_type = data_Target[data_Target[:, 0] == target[0], 3]
            x, y = original_position[0]
            plt.plot(x, y, 'o', color=[0.5, 0.5, 0.5], markerfacecolor='g')
            plt.text(x + 0.01, y, str(target[0]), visible=True)
        plt.grid(True)
        plt.show()

        for target in cluster_1[1]:
            original_position = data_Target[data_Target[:, 0] == target[0], 1:3]
            x, y = original_position[0]
            angle = calculate_angle(x, y)
            target[-2] = angle

        num_clusters = 3
        best_solution, best_value, best_assignment = Particle_algorithm.pso_optimization(targets, num_clusters)

        print("最佳分配方案:", best_assignment)

        end_time = time.time()
        execution_time = end_time - start_time
        print(f"PSO分簇耗时: {execution_time:.3f} 秒")

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
            z = cluster_targets[:, 3]  
            indices = cluster_targets[:, 0]

            ax.scatter(x, y, z, color=colors[cluster_id], label=f'Cluster {cluster_id + 1}')

            for j in range(len(x)):
                ax.text(x[j], y[j], z[j], f'{int(indices[j])}', color='black')

        ax.set_xlabel('X Position')
        ax.set_ylabel('Y Position')
        ax.set_zlabel('Value')
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

    else:
        print(f"New target {new_target_index} not added to destroy targets set.")

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
