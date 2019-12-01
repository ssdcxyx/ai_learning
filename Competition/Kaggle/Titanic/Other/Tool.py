# -*- coding: utf-8 -*-
# @Time    : 2019-06-30 09:36
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Other.py


class Entry:
    def __init__(self, label, value):
        self.label = label
        self.value = value


# 特征组合
def feature_combination(features, i, combinations):
    if i == len(features):
        return combinations
    temp = []
    if i == 0:
        for feature in features[i]:
            temp.append([feature])
    else:
        for combination in combinations:
            for feature in features[i]:
                combination_copy = combination.copy()
                combination_copy.append(feature)
                temp.append(combination_copy)
    i += 1
    return feature_combination(features, i, temp)





