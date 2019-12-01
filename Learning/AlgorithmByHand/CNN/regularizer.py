# -*- coding: utf-8 -*-
# @time       : 7/11/2019 10:13 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : regularizer.py
# @description: 
import numpy as np


def norm1(W):
    return np.sum(np.abs(W))


def norm1_backward(W):
    return np.sign(W)


def norm2(W):
    return np.sum(W * W)


def norm2_backward(W):
    return W
