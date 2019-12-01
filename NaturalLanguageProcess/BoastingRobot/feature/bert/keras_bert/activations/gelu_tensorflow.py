# -*- coding: utf-8 -*-
# @time       : 15/11/2019 10:23 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : gelu_tensorflow
# @description: 


from tensorflow.python.ops.math_ops import erf, sqrt

__all__ = ['gelu']


def gelu(x):
    return 0.5 * x * (1.0 + erf(x / sqrt(2.0)))
