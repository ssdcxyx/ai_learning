# -*- coding: utf-8 -*-
# @time       : 15/11/2019 10:28 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : inputs
# @description: 


from feature.bert.keras_bert.backend import keras


def get_inputs(seq_len):
    """Get input layers.
    See: https://arxiv.org/pdf/1810.04805.pdf
    :param seq_len: Length of the sequence or None.
    """
    names = ['Token', 'Segment', 'Masked']
    return [keras.layers.Input(
        shape=(seq_len,),
        name='Input-%s' % name,
    ) for name in names]