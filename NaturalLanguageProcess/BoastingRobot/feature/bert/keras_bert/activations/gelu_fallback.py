# -*- coding: utf-8 -*-
# @time       : 15/11/2019 10:21 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : gelu_fallback
# @description: 


import math
from feature.bert.keras_bert.backend import backend as K

__all__ = ['gelu']


def gelu(x):
    return 0.5 * x * (1.0 + K.tanh(math.sqrt(2.0 / math.pi) * (x + 0.044715 * x * x * x)))