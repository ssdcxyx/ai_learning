# -*- coding: utf-8 -*-
# @time       : 10/11/2019 10:25 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : layers.py
# @description: 


import numpy as np
from Learning.AlgorithmByHand.CNN import activations
from Learning.AlgorithmByHand.CNN import regularizer

import matplotlib.pyplot as plt
import cv2


# 卷积层
class Conv2D:

    def __init__(self,
                 inputs,
                 filters,
                 kernel_size,
                 strides=(1, 1),
                 padding='valid',
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 name=None):
        self.inputs = inputs
        self.filters = filters
        self.kernel_size = kernel_size
        self.strides = strides
        self.padding = padding
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.name =name
        self.inputs_pad = None
        self.W, self.b = None, None
        self.A = None, None
        self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev = None, None, None, None
        self.h_pad, self.w_pad = None, None
        self.n_H, self.n_W, self.n_C = None, None, None
        self.vdw, self.vdb, self.sdw, self.sdb, self.t = 0, 0, 0, 0, 0
        self.init_parameters()

    def init_parameters(self):
        # 参数初始化
        (self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev) = self.inputs.A.shape
        if self.padding == 'valid':
            self.n_H = int(np.ceil((self.n_H_prev - self.kernel_size[0] + 1) / self.strides[0]))
            self.n_W = int(np.ceil((self.n_W_prev - self.kernel_size[1] + 1) / self.strides[1]))
        elif self.padding == "same":
            self.n_H = int(np.ceil(self.n_H_prev / self.strides[0]))
            self.n_W = int(np.ceil(self.n_W_prev // self.strides[1]))
            self.h_pad = int(max((self.n_H - 1) * self.strides[0] + self.kernel_size[0] - self.n_H_prev, 0))
            self.w_pad = int(max((self.n_W - 1) * self.strides[1] + self.kernel_size[1] - self.n_W_prev, 0))
        self.n_C = self.filters
        if self.kernel_initializer == "he_init":
            self.W = np.random.normal(loc=0,
                                      scale=np.sqrt(1.0/((self.n_H_prev*self.n_W_prev*self.n_C_prev+self.n_H*self.n_W*self.filters)/2.0)),
                                      size=(self.kernel_size[0], self.kernel_size[1], self.n_C_prev, self.filters))
        else:
            self.W = np.zeros((self.kernel_size[0], self.kernel_size[1], self.n_C_prev, self.filters))
        self.b = np.zeros((1, 1, 1, self.filters))

        # 正则化参数
        if self.kernel_regularizer == "l1":
            self.kernel_regularizer = regularizer.norm1
        elif self.kernel_regularizer == "l2":
            self.kernel_regularizer = regularizer.norm2

        self.A = np.zeros((self.m, self.n_H, self.n_W, self.filters))

    def forward(self):
        def conv_single_step(a_slice_prev, W, b):
            s = a_slice_prev * W
            A = np.sum(s)
            A = np.squeeze(np.add(A, b))
            return A

        def zero_pad(X, h_pad, w_pad):
            X_pad = np.pad(X, ((0, 0), (h_pad//2, h_pad-h_pad//2), (w_pad//2, w_pad-w_pad//2), (0, 0)),
                           mode='constant', constant_values=(0, 0))
            return X_pad

        (self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev) = self.inputs.A.shape
        self.inputs_pad = zero_pad(self.inputs.A, self.h_pad, self.w_pad)
        self.A = np.zeros((self.m, self.n_H, self.n_W, self.filters))

        for i in range(self.m):
            a_prev_pad = self.inputs_pad[i, :, :, :]
            for h in range(self.n_H):
                vert_start = h * self.strides[0]
                vert_end = vert_start + self.kernel_size[0]
                for w in range(self.n_W):
                    horiz_start = w * self.strides[1]
                    horiz_end = horiz_start + self.kernel_size[1]
                    for c in range(self.n_C):
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        self.A[i, h, w, c] = conv_single_step(a_slice_prev, self.W[:, :, :, c], self.b[:, :, :, c])
        # 显示卷积
        # for c in range(self.n_C):
        #     plt.subplot(1, self.n_C, c+1)
        #     plt.imshow(self.A[0, :, :, c])
        # plt.show()
        # cv2.waitKey()

    def backward(self, dA, lr, lambd=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        def zero_pad(X, h_pad, w_pad):
            X_pad = np.pad(X, ((0, 0), (h_pad//2, h_pad-h_pad//2), (w_pad//2, w_pad-w_pad//2), (0, 0)),
                           mode='constant', constant_values=(0, 0))
            return X_pad
        dW = np.zeros((self.kernel_size[0], self.kernel_size[1], self.n_C_prev, self.filters))
        db = np.zeros((1, 1, 1, self.filters))
        dA_prev = np.zeros((self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev))
        dA_prev_pad = zero_pad(dA_prev, self.h_pad, self.w_pad)
        for i in range(self.m):
            a_prev_pad = self.inputs_pad[i, :, :, :]
            da_prev_pad = dA_prev_pad[i, :, :, :]
            for h in range(self.n_H):
                vert_start = h * self.strides[0]
                vert_end = vert_start + self.kernel_size[0]
                for w in range(self.n_W):
                    horiz_start = w * self.strides[1]
                    horiz_end = horiz_start + self.kernel_size[1]
                    for c in range(self.n_C):
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.W[:, :, :, c] * dA[i, h, w, c]

                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        dW[:, :, :, c] += a_slice * dA[i, h, w, c]
                        db[:, :, :, c] += dA[i, h, w, c]

            if self.h_pad != 0 and self.w_pad != 0:
                dA_prev[i, :, :, :] = da_prev_pad[self.h_pad//2:-(self.h_pad - self.h_pad // 2), self.w_pad//2:-(self.w_pad - self.w_pad//2), :]
            elif self.h_pad != 0:
                dA_prev[i, :, :, :] = da_prev_pad[self.h_pad // 2:-(self.h_pad - self.h_pad // 2), :]
            elif self.w_pad != 0:
                dA_prev[i, :, :, :] = da_prev_pad[:, self.w_pad//2:-(self.w_pad - self.w_pad // 2), :]

        # Adam
        self.t += 1
        self.vdw = (beta1 * self.vdw + (1 - beta1) * dW) / (1 - beta1 ** self.t)
        self.vdb = (beta1 * self.vdb + (1 - beta1) * db) / (1 - beta1 ** self.t)
        self.sdw = (beta2 * self.sdw + (1 - beta2) * dW * dW) / (1 - beta2 ** self.t)
        self.sdb = (beta2 * self.sdb + (1 - beta2) * db * db) / (1 - beta2 ** self.t)
        dW = self.vdw / (self.sdw ** 0.5 + epsilon)
        db = self.vdb / (self.sdb ** 0.5 + epsilon)
        self.W -= lr * (dW / self.m + self.kernel_regularizer(self.W) * lambd / self.m)
        self.b -= lr * db / self.m
        return dA_prev


# 池化层
class Pooling2D:

    def __init__(self,
                 inputs,
                 pool_size,
                 strides=(1, 1),
                 padding='valid',
                 mode="max",
                 name=None):
        self.inputs = inputs
        self.pool_size = pool_size
        self.strides = strides
        self.padding = padding
        self.mode = mode
        self.name = name
        self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev = None, None, None, None
        self.h_pad, self.w_pad = None, None
        self.n_H, self.n_W, self.n_C = None, None, None
        self.A = None
        self.init_parameters()

    def init_parameters(self):
        # 参数初始化
        (self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev) = self.inputs.A.shape
        if self.padding == 'valid':
            self.n_H = int(np.ceil((self.n_H_prev - self.pool_size[0] + 1) / self.strides[0]))
            self.n_W = int(np.ceil((self.n_W_prev - self.pool_size[1] + 1) / self.strides[1]))
        elif self.padding == "same":
            self.n_H = int(np.ceil(self.n_H_prev / self.strides[0]))
            self.n_W = int(np.ceil(self.n_W_prev // self.strides[1]))
            self.h_pad = int(max((self.n_H - 1) * self.strides[0] + self.pool_size[0] - self.n_H_prev, 0))
            self.w_pad = int(max((self.n_W - 1) * self.strides[1] + self.pool_size[1] - self.n_W_prev, 0))
        self.A = np.zeros((self.m, self.n_H, self.n_W, self.n_C_prev))

    def forward(self):
        def zero_pad(X, h_pad, w_pad):
            X_pad = np.pad(X, ((0, 0), (h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0)),
                           mode='constant', constant_values=(0, 0))
            return X_pad

        (self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev) = self.inputs.A.shape
        self.n_C = self.n_C_prev
        self.inputs_pad = zero_pad(self.inputs.A, self.h_pad, self.w_pad)
        self.A = np.zeros((self.m, self.n_H, self.n_W, self.n_C_prev))
        for i in range(self.m):
            for h in range(self.n_H):
                vert_start = h * self.strides[0]
                vert_end = vert_start + self.pool_size[0]
                for w in range(self.n_W):
                    horiz_start = w * self.strides[1]
                    horiz_end = horiz_start + self.pool_size[1]
                    for c in range(self.n_C):
                        a_prev_slice = self.inputs.A[i, vert_start:vert_end, horiz_start:horiz_end, c]
                        if self.mode == "max":
                            self.A[i, h, w, c] = np.max(a_prev_slice)
                        elif self.mode == "avg":
                            self.A[i, h, w, c] = np.mean(a_prev_slice)

        # 显示池化
        # for c in range(self.n_C):
        #     plt.subplot(1, self.n_C, c+1)
        #     plt.imshow(self.A[0, :, :, c])
        # plt.show()
        # cv2.waitKey()

    def backward(self, dA):

        def zero_pad(X, h_pad, w_pad):
            X_pad = np.pad(X, ((0, 0), (h_pad // 2, h_pad - h_pad // 2), (w_pad // 2, w_pad - w_pad // 2), (0, 0)),
                           mode='constant', constant_values=(0, 0))
            return X_pad

        def create_mask_from_windows(X):
            mask = X == np.max(X)
            return mask

        def distribute_value(dZ, shape):
            (n_H, n_W) = shape
            average = dZ / (n_H * n_W)
            a = np.ones((n_H, n_W)) * average
            return a
        dZ = dA
        dA_prev = np.zeros((self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev))
        dA_prev_pad = zero_pad(dA_prev, self.h_pad, self.w_pad)
        for i in range(self.m):
            a_prev_pad = self.inputs_pad[i, :, :, :]
            da_prev = dA_prev_pad[i, :, :, :]
            for h in range(self.n_H):
                vert_start = h * self.strides[0]
                vert_end = vert_start + self.pool_size[0]
                for w in range(self.n_W):
                    horiz_start = w * self.strides[1]
                    horiz_end = horiz_start + self.pool_size[1]
                    for c in range(self.n_C):
                        if self.mode == "max":
                            a_prev_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, c]
                            mask = create_mask_from_windows(a_prev_slice)
                            da_prev[i, vert_start:vert_end, horiz_start:horiz_end, c] += mask * dA[i, w, h, c]
                        elif self.mode == "avg":
                            dz = dZ[i, h, w, c]
                            if distribute_value(dz, (self.pool_size[0], self.pool_size[1])).shape != (5, 5):
                                print()
                            if da_prev[vert_start:vert_end, horiz_start:horiz_end, c].shape != (5, 5):
                                print()
                            da_prev[vert_start:vert_end, horiz_start:horiz_end, c] += distribute_value(dz, (self.pool_size[0], self.pool_size[1]))

            if self.h_pad != 0 and self.w_pad != 0:
                dA_prev[i, :, :, :] = da_prev[self.h_pad // 2:-(self.h_pad - self.h_pad // 2),
                                      self.w_pad // 2:-(self.w_pad - self.w_pad // 2), :]
            elif self.h_pad != 0:
                dA_prev[i, :, :, :] = da_prev[self.h_pad // 2:-(self.h_pad - self.h_pad // 2), :]
            elif self.w_pad != 0:
                dA_prev[i, :, :, :] = da_prev[self.w_pad // 2:-(self.w_pad - self.w_pad // 2), :]
        return dA_prev


class Dense:

    def __init__(self,
                 inputs,
                 units,
                 kernel_initializer=None,
                 kernel_regularizer=None,
                 name=None):
        self.inputs = inputs
        self.units = units
        self.kernel_initializer = kernel_initializer
        self.kernel_regularizer = kernel_regularizer
        self.name = name
        self.W, self.b = None, None
        self.A = None
        self.m, self.n = None, None
        self.vdw, self.vdb, self.sdw, self.sdb, self.t = 0, 0, 0, 0, 0
        self.init_parameters()

    def init_parameters(self):
        (self.m, self.n) = self.inputs.A.shape
        # 参数初始化
        if self.kernel_initializer == 'he_init':
            self.W = np.random.normal(loc=0,
                                      scale=np.sqrt(1.0/((self.units + self.n)/2.0)),
                                      size=(self.n, self.units))
        else:
            self.W = np.zeros((self.n, self.units))
        self.b = np.zeros((1, self.units))
        # 正则化参数
        if self.kernel_regularizer == "l1":
            self.kernel_regularizer = regularizer.norm1_backward
        elif self.kernel_regularizer == "l2":
            self.kernel_regularizer = regularizer.norm2_backward

        self.A = np.zeros((self.m, self.units))

    def forward(self):
        (self.m, self.n) = self.inputs.A.shape
        self.A = np.zeros((self.m, self.units))
        # 前向传播
        self.A = np.dot(self.inputs.A, self.W) + self.b

    def backward(self, dA, lr, lambd=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        # 反向传播
        self.t += 1
        dA_prev, dW, db = activations.linear_backward(dA, self.inputs.A, self.W)
        self.vdw = (beta1 * self.vdw + (1 - beta1) * dW) / (1 - beta1 ** self.t)
        self.vdb = (beta1 * self.vdb + (1 - beta1) * db) / (1 - beta1 ** self.t)
        self.sdw = (beta2 * self.sdw + (1 - beta2) * dW * dW) / (1 - beta2 ** self.t)
        self.sdb = (beta2 * self.sdb + (1 - beta2) * db * db) / (1 - beta2 ** self.t)
        dW = self.vdw / (self.sdw ** 0.5 + epsilon)
        db = self.vdb / (self.sdb ** 0.5 + epsilon)
        self.W -= lr * (dW / self.m + self.kernel_regularizer(self.W) * lambd / self.m)
        self.b -= lr * db / self.m
        return dA_prev


class Flatten:

    def __init__(self,
                 inputs,
                 name=None):
        self.inputs = inputs
        self.A = None
        self.name = name
        self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev = None, None, None, None
        self.init_parameters()

    def init_parameters(self):
        (self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev) = self.inputs.A.shape
        self.A = self.inputs.A.reshape((self.inputs.A.shape[0], -1))

    def forward(self):
        (self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev) = self.inputs.A.shape
        self.A = self.inputs.A.reshape((self.inputs.A.shape[0], -1))

    def backward(self, dA):
        dA_prev = dA.reshape(self.m, self.n_H_prev, self.n_W_prev, self.n_C_prev)
        return dA_prev


class Dropout:

    def __init__(self,
                 inputs,
                 rate=0.0,
                 name=None):
        self.inputs = inputs
        self.rate = rate
        self.name = name
        self.A = None
        self.m, self.n = None, None
        self.drop = None
        self.init_parameter()

    def init_parameter(self):
        (self.m, self.n) = self.inputs.A.shape
        self.A = np.zeros((self.m, self.n))

    def forward(self, is_train):
        (self.m, self.n) = self.inputs.A.shape
        self.A = np.zeros((self.m, self.n))
        if is_train:
            self.drop = np.random.binomial(1, 1 - self.rate,
                                           self.m * self.n).reshape(self.inputs.A.shape)
            self.A = self.inputs.A * self.drop * 2
        else:
            self.A = self.inputs.A

    def backward(self, dA, is_train):
        if is_train:
            dA_prev = dA * self.rate
        else:
            dA_prev = dA
        return dA_prev


class Input:
    def __init__(self,
                 inputs,
                 name):
        self.A = inputs
        self.name = name

    def init_parameters(self):
        pass

    def forward(self, inputs):
        self.A = inputs

    def backward(self):
        pass


class Activation:

    def __init__(self,
                 inputs,
                 activation=None,
                 name=None):
        self.inputs = inputs
        self.activation = activation
        self.name = name
        self.A = None
        self.activation_backward = None
        self.init_parameter()

    def init_parameter(self):
        # 激活函数
        if self.activation == "relu":
            self.activation = activations.relu
            self.activation_backward = activations.relu_backward
        elif self.activation == "sigmoid":
            self.activation = activations.sigmoid
            self.activation_backward = activations.sigmoid_backward
        elif self.activation == "softmax":
            self.activation = activations.softmax
        else:
            self.activation = activations.linear
            self.activation_backward = activations.linear_backward
        self.A = np.zeros(self.inputs.A.shape)

    def forward(self):
        self.A = self.activation(self.inputs.A)

    def backward(self, dA):
        dA_prev = dA
        if self.activation_backward is not None:
            if self.activation is not activations.linear:
                dA_prev = self.activation_backward(dA, self.A)
        return dA_prev


class BatchNormalization:

    def __init__(self,
                 inputs,
                 name=None):
        self.inputs = inputs
        self.name = name
        self.A = None
        self.init_parameter()

    def init_parameter(self):
        self.A = np.zeros(self.inputs.A.shape)

    def forward(self):
        batch_mean = self.inputs.A.sum(axis=0) / self.inputs.A.shape[0]
        batch_var = ((self.inputs.A - batch_mean) ** 2).sum(axis=0) / self.inputs.A.shape[0]
        self.A = (self.inputs.A - batch_mean) / ((batch_var + 1e-8) ** 0.5)

    def backward(self, dA):
        return dA



