import numpy as np
    
def TextOutput(Distance = None,Demand = None,route = None,Capacity = None): 
    ## 输出�径函�
#输入：route ��
#输出：p �径文�形式
    
    ## 总路�
    len_ = len(route)
    
    print('Best Route:')
    p = num2str(route(1))
    
    for i in np.arange(2,len_+1).reshape(-1):
        p = np.array([p,' -> ',num2str(route(i))])
    
    print(p)
    ## 子路�
    
    route = route + 1
    
    Vnum = 1
    
    DisTraveled = 0
    
    delivery = 0
    
    subpath = '0'
    
    for j in np.arange(2,len_+1).reshape(-1):
        DisTraveled = DisTraveled + Distance(route(j - 1),route(j))
        delivery = delivery + Demand(route(j))
        subpath = np.array([subpath,' -> ',num2str(route(j) - 1)])
        if route(j) == 1:
            print('-------------------------------------------------------------')
            print('Route of Vehichle No.%d: %s  \n' % (Vnum,subpath))
            print('Distance traveled: %.2f km, load rate: %.2f%%;  \n' % (DisTraveled,delivery / Capacity * 100))
            Vnum = Vnum + 1
            DisTraveled = 0
            delivery = 0
            subpath = '0'
    