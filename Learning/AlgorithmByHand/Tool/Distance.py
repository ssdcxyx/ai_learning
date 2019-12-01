# -*- coding: utf-8 -*-
# @time       : 4/11/2019 10:33 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : Distance.py
# @description: 


def get_euclidean_distance(arr1, arr2):
    return sum((x1 - x2) ** 2 for x1, x2 in zip(arr1, arr2)) ** 0.5


def get_cosine_distance(arr1, arr2):
    numerator = sum(x1 * x2 for x1, x2 in zip(arr1, arr2))
    denominator = (sum(x1 ** 2 for x1 in arr1) * sum(x2 ** 2 for x2 in arr2)) ** 0.5
    return numerator / denominator