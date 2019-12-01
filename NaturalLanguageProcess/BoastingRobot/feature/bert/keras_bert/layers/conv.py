# -*- coding: utf-8 -*-
# @time       : 15/11/2019 10:27 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : conv.py
# @description: 


from feature.bert.keras_bert.backend import keras
from feature.bert.keras_bert.backend import backend as K


class MaskedConv1D(keras.layers.Conv1D):

    def __init__(self, **kwargs):
        super(MaskedConv1D, self).__init__(**kwargs)
        self.supports_masking = True

    def compute_mask(self, inputs, mask=None):
        if mask is not None and self.padding == 'valid':
            mask = mask[:, self.kernel_size[0] // 2 * self.dilation_rate[0] * 2:]
        return mask

    def call(self, inputs, mask=None):
        if mask is not None:
            mask = K.cast(mask, K.floatx())
            inputs *= K.expand_dims(mask, axis=-1)
        return super(MaskedConv1D, self).call(inputs)