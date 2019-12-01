# -*- coding: utf-8 -*-
# @time       : 15/11/2019 10:22 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : gelu_selection
# @description: 


from feature.bert.keras_bert.backend import backend as K

__all__ = ['gelu']

if K.backend() == 'tensorflow':
    from .gelu_tensorflow import gelu
else:
    from .gelu_fallback import gelu
