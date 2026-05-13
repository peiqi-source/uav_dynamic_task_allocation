"""ÀúÊ·°æ±¾ÖÐµÄtextÊä³ö½Å±¾£¬±£ÁôÓÃÓÚËã·¨¶ÔÕÕ¡¢¸´ÏÖÊµÑé»òÇ¨ÒÆ²Î¿¼¡£"""
import numpy as np

def TextOutput(Distance = None,Demand = None,route = None,Capacity = None):
    ## è¾“å‡ºè·å¾„å‡½æ•
#è¾“å…¥ï¼šroute è·å¾
#è¾“å‡ºï¼šp è·å¾„æ–‡æœå½¢å¼

    ## æ€»è·¯å¾
    len_ = len(route)

    print('Best Route:')
    p = num2str(route(1))

    for i in np.arange(2,len_+1).reshape(-1):
        p = np.array([p,' -> ',num2str(route(i))])

    print(p)
    ## å­è·¯å¾

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
