# -*- coding: utf-8 -*-
# @time       : 5/11/2019 8:49 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : activations.py
# @description: 

import numpy as np


def sigmoid(Z):
    A = 1 / (1 + np.exp(-Z))
    return A


def sigmoid_backward(dA, A):
    dZ = dA * A * (1 - A)
    return dZ


def relu(Z):
    A = np.maximum(0, Z)
    return A


def relu_backward(dA, A):
    dZ = np.array(dA, copy=True)
    dZ[A == 0] = 0
    return dZ


def softmax(Z):
    A = np.zeros(Z.shape)
    # A = np.exp(Z)/np.sum(np.exp(Z))
    # 避免nan
    for i in range(A.shape[0]):
        A[i, :] = np.exp(Z[i, :] - np.max(Z[i, :])) / np.sum(np.exp(Z[i, :] - np.max(Z[i, :])))
    return A


def linear(Z):
    A = Z
    return A


def linear_backward(dZ, A_prev, W):
    dW = np.dot(A_prev.T, dZ)
    db = np.sum(dZ, axis=0, keepdims=True)
    dA_prev = np.dot(dZ, W.T)
    return dA_prev, dW, db


def get(identifier):
    if identifier is None:
        return linear
