# -*- coding: utf-8 -*-
# @time       : 4/11/2019 9:21 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : Preprocess.py
# @description: 

from random import seed, random


def load_data(data_path):
    f = open(data_path)
    X = []
    y = []
    for line in f:
        line = line[:-1].split(',')
        xi = [float(s) for s in line[:-1]]
        yi = line[-1]
        if '.' in yi:
            yi = float(yi)
        else:
            yi = int(yi)
        X.append(xi)
        y.append(yi)
    f.close()
    return X, y


def train_test_split(X, y, test_size=0.3, random_state=None):
    if random_state is not None:
        seed(random_state)
    X_train = []
    X_test = []
    y_train = []
    y_test = []
    for i in range(len(X)):
        if random() < 1 - test_size:
            X_train.append(X[i])
            y_train.append(y[i])
        else:
            X_test.append(X[i])
            y_test.append(y[i])
    return X_train, y_train, X_test, y_test


def min_max_scale(X):
    m = len(X[0])
    x_max = [-float('inf') for _ in range(m)]
    x_min = [float('inf') for _ in range(m)]
    for row in X:
        x_max = [max(a, b) for a, b in zip(x_max, row)]
        x_min = [min(a, b) for a, b in zip(x_min, row)]
    ret = []
    for row in X:
        tmp = [(x-b)/(a-b) for a, b, x in zip(x_max, x_min, row)]
        ret.append(tmp)
    return ret