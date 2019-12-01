# -*- coding: utf-8 -*-
# @Time    : 2019-05-17 19:49
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : tool.py

import math
import numpy as np


# 获得单位圆分位点坐标
def twevle_quantile_of_unit_circle(num):
    if num == 0:
        return [0, 1]
    elif num == 1:
        return [1 / 2, math.sqrt(3) / 2]
    elif num == 2:
        return [math.sqrt(3) / 2, 1 / 2]
    elif num == 3:
        return [1, 0]
    elif num == 4:
        return [math.sqrt(3) / 2, -1 / 2]
    elif num == 5:
        return [1 / 2, -math.sqrt(3) / 2]
    elif num == 6:
        return [0, -1]
    elif num == 7:
        return [-1 / 2, -math.sqrt(3) / 2]
    elif num == 8:
        return [-math.sqrt(3) / 2, -1 / 2]
    elif num == 9:
        return [-1, 0]
    elif num == 10:
        return [-math.sqrt(3) / 2, 1 / 2]
    elif num == 11:
        return [-1 / 2, math.sqrt(3) / 2]


# 抽样
def sample(data, sample_size=1):
    sample = np.random.permutation(len(data))
    sample_data =  data.take(sample[:sample_size])
    return sample_data

