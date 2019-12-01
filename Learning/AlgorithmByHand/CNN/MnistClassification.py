# -*- coding: utf-8 -*-
# @time       : 9/11/2019 9:21 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : MnistClassification.py
# @description: 


from Learning.AlgorithmByHand.CNN.layers import Input, Conv2D, Pooling2D, Dense, Flatten, Dropout, Activation, BatchNormalization
from Learning.AlgorithmByHand.CNN.regularizer import norm1, norm2

import cv2
import glob
import os
import numpy as np
from random import sample


def load_data(image_path, label_path):
    with open(image_path, 'rb') as f1:
        image_file = f1.read()
    with open(label_path, 'rb') as f2:
        label_file = f2.read()
    image_file = image_file[16:]
    label_file = label_file[8:]
    X_train, y_train = [], []
    for i in range(100):
        label = int(label_file[i])
        image_list = [int(item/255) for item in image_file[i * 784:i * 784 + 784]]
        image_np = np.array(image_list, dtype=np.uint8).reshape(28, 28, 1)
        # cv2.imshow('1', image_np)
        # cv2.waitKey()
        X_train.append(list(image_np)), y_train.append(label)
    return np.array(X_train), np.array(y_train)


def cross_entropy(outputs, y, Ws, regularizer="l2", lambd=0.001):
    m = outputs.shape[0]
    loss = np.sum(-np.log(outputs[range(m), y])) / m
    if regularizer == "l2":
        for W in Ws:
            loss += norm2(W) * lambd / (2 * m)
    elif regularizer == "l1":
        for W in Ws:
            loss += norm1(W) * lambd / m
    return loss


def get_acc(outputs, y):
    outputs = np.argmax(outputs, axis=1)
    return sum(outputi == yi for outputi, yi in zip(outputs, y)) / len(y) * 100


def delta_cross_entropy(outputs, y):
    m = outputs.shape[0]
    grad = outputs.copy()
    grad[range(m), y] -= 1
    return grad


def model(X_train, y_train, epochs=100, batch_size=8, learning_rate=0.001, dropout=0.25):
    (m, H, W, C) = X_train.shape
    init = True
    _layers = []
    conv1, bn1, relu1, pool1, flatten1, dropout1, dense1, relu2, dense2, output_layer = None, None, None, None, None, None, None, None, None, None
    for epoch in range(epochs):
        rnd_idx = np.random.permutation(len(X_train))
        iters = len(X_train) // batch_size
        for batch_idx in np.array_split(rnd_idx, iters):
            X_batch, y_batch = X_train[batch_idx], y_train[batch_idx]
            if init:
                input_layer = Input(inputs=X_batch, name="input")
                conv1 = Conv2D(inputs=input_layer, filters=5, kernel_size=(5, 5), strides=(1, 1), padding="same",
                               kernel_initializer="he_init", kernel_regularizer="l2", name="conv1")
                bn1 = BatchNormalization(inputs=conv1, name="bn1")
                relu1 = Activation(inputs=bn1, activation="relu", name="relu1")
                pool1 = Pooling2D(inputs=relu1, pool_size=(5, 5), strides=(2, 2), padding="same", mode="avg",
                                  name="pool1")
                # 展平
                flatten1 = Flatten(inputs=pool1, name="flutter")
                # Dropout
                dropout1 = Dropout(inputs=flatten1, rate=dropout, name="dropout1")
                dense1 = Dense(inputs=dropout1, units=100, kernel_initializer="he_init",
                               kernel_regularizer="l2", name="dense1")
                relu2 = Activation(inputs=dense1, activation="relu", name="relu2")
                dense2 = Dense(inputs=relu2, units=10, kernel_initializer="he_init",
                               kernel_regularizer="l2", name="dense2")
                output_layer = Activation(inputs=dense2, activation="softmax", name="output")
                Ws = [conv1.W, dense1.W, dense2.W]
                init = False
            conv1.inputs = Input(inputs=X_batch, name="input")
            conv1.forward()
            bn1.forward()
            relu1.forward()
            pool1.forward()
            flatten1.forward()
            dropout1.forward(is_train=True)
            dense1.forward()
            relu2.forward()
            dense2.forward()
            output_layer.forward()
            # batch loss
            # print("epoch:", epoch, "\tloss:", cross_entropy(output_layer.A, y_batch, Ws, regularizer="l2", l=l))
            dA = delta_cross_entropy(output_layer.A, y_batch)
            dA = dense2.backward(dA, lr=learning_rate)
            dA = relu2.backward(dA)
            dA = dense1.backward(dA, lr=learning_rate)
            dA = dropout1.backward(dA, is_train=True)
            dA = flatten1.backward(dA)
            dA = pool1.backward(dA)
            dA = relu1.backward(dA)
            dA = bn1.backward(dA)
            dA = conv1.backward(dA, lr=learning_rate)
        conv1.inputs = Input(inputs=X_train, name="input")
        conv1.forward()
        bn1.forward()
        relu1.forward()
        pool1.forward()
        flatten1.forward()
        dropout1.forward(is_train=True)
        dense1.forward()
        relu2.forward()
        dense2.forward()
        output_layer.forward()
        print("epoch:", epoch, "\tloss:", cross_entropy(output_layer.A, y_train, Ws, regularizer="l2"),
              "\taccuracy:", get_acc(output_layer.A, y_train), "%")


if __name__ == '__main__':
    X_train, y_train = load_data("mnist/train-images-idx3-ubyte", "mnist/train-labels-idx1-ubyte")
    model(X_train, y_train)
    print()



