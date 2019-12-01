# -*- coding: utf-8 -*-
# @Time    : 2019-01-14 17:52
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : DeepDream.py


import os
from io import BytesIO
import numpy as np
from functools import partial
import PIL.Image

import tensorflow as tf

# 获取预训练模型
# wget https://storgage.googleapis.com/download.tensorflow.org/models/inception5h.zip && unzip inception5h.zip

model_fn = 'tensorflow_inception_graph.pb'

graph = tf.Graph()
sess = tf.InteractiveSession(graph=graph)
with tf.gfile.FastGFile(model_fn, 'rb') as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())
t_input = tf.placeholder(np.float32, name="input")
imagenet_mean = 117.0
t_preprocessed = tf.expand_dims(t_input-imagenet_mean, 0)
tf.import_graph_def(graph_def, {'input': t_preprocessed})

layers = [op.name for op in graph.get_operations() if op.type == "Conv2D" and "import/" in op.name]
feature_nums = [int(graph.get_tensor_by_name(name+":0").get_shape()[-1]) for name in layers]

print("Number of layers", len(layers))
print("Total number of feature channels:", sum(feature_nums))




