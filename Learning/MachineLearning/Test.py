# -*- coding: utf-8 -*-
# @Time    : 2018-12-03 15:57
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Test.py


from sklearn.preprocessing import StandardScaler

data = [[0, 0], [1, 0], [2, 1], [3, 1]]
scaler = StandardScaler()
scaler.fit(data)

scaler_data = scaler.transform(data)
print()