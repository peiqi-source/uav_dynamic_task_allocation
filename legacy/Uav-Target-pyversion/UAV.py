import numpy as np
import UAV_Value
import Kmeans_UAV

def min_max_normalization(data):
    min_value = min(data)
    max_value = max(data)
    normalized_data = [(value - min_value) / (max_value - min_value) for value in data]
    return normalized_data


def UAV(data_UAV = None):

    attack = 0
    
    test = data_UAV
    
    m, n = test.shape    # 获取列表的长和宽 m = 12, n = 6
    
    # 数据归一化�理
    # data.num = test(:,1)
    # data.x = np.transpose(mapminmax(np.transpose(test(:,2)),0,1))
    # data.y = np.transpose(mapminmax(np.transpose(test(:,3)),0,1))
    # data.type = test(:,4)
    # data.distance = np.transpose(mapminmax(np.transpose(test(:,5)),0,1))
    # data.attack = test(:,6)

    class data_pre:
        def __init__(self):
            self.num = ""
            self.x = ""
            self.y = ""
            self.type = ""
            self.distance = ""
            self.attack = ""

    np_data_uav = np.array(data_UAV)
    # print(np_data_UAV[:, 0].tolist())

    data = data_pre()
    data.num = np_data_uav[:, 0].tolist()
    data.x = min_max_normalization(np_data_uav[:, 1].tolist())
    data.y = min_max_normalization(np_data_uav[:, 2].tolist())
    data.type = np_data_uav[:, 3]
    data.distance = min_max_normalization(np_data_uav[:,4].tolist())
    data.attack = np_data_uav[:,5]

    # 调用价值函数，计算出导引机和其他无人机之间的价值
    value, d, data_UAV, test1 = UAV_Value.UAV_Value(data_UAV, m, n)
    cluster_centers, cluster_results = Kmeans_UAV.Kmeans_UAV(test1, d, data_UAV)
    print(cluster_results)

    """
    #     # 拍卖攻击机，导引机是竞标�
#     # UAV_allocation = UAV_Auction(value);
    
    #     for i = 1: d
#         UAV_allocation_dao =UAV_allocation{i,1};   # 存储导引机信�
#         UAV_allocation_gen =cell2mat(UAV_allocation{i,2}); # 存储跟随机信�
    
    #         for k = 1:length(test(:,1))
    
    #             if UAV_allocation_dao(1,1) == test(k,1)     # 获取导引机的�有信�
#                 UAV_allocation1(1,:) = test(k,2:6);
#             end
    
    #             for j = 1:length(UAV_allocation_gen(:,1))   # 获取跟随机所有信�
#                 if UAV_allocation_gen(j,1) == test(k,1)
#                     UAV_allocation2(j,:) = test(k,2:6);
#                 end
#             end
    
    #         end
    
    #         UAV_allocation_dao = [UAV_allocation_dao,UAV_allocation1];
#         UAV_allocation_gen = [UAV_allocation_gen,UAV_allocation2];
#         UAV_allocation_result{i,1} = [UAV_allocation_dao;UAV_allocation_gen];
    
    #         # 计算每一�无人机簇的�弹�
#         UAV_allocation_result{i,2} = UAV_allocation_dao(1,6)+sum(UAV_allocation_gen(:,6));
#         attack = attack + UAV_allocation_result{i,2};
    
    #         clear UAV_allocation_dao;
#         clear UAV_allocation2;
#         clear UAV_allocation_gen;
#     end
    """

    return d, attack, cluster_results, test1
