# -*- coding: utf-8 -*-
# @time       : 15/11/2019 10:26 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : __init__.py
# @description: 


from .inputs import get_inputs
from .embedding import get_embedding, TokenEmbedding, EmbeddingSimilarity
from .masked import Masked
from .extract import Extract
from .pooling import MaskedGlobalMaxPool1D
from .conv import MaskedConv1D
from .task_embed import TaskEmbedding
