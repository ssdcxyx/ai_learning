# -*- coding: utf-8 -*-
# @time       : 3/11/2019 12:06 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : BoF.py
# @description: 


import glob
import os
from random import random, seed, randint

import numpy as np
import cv2
from sklearn.cluster import KMeans
import joblib
import matplotlib.pyplot as plt

# from ComputerVision.BoF.KMeans import KMeans

IMG_PATH = "./img/"
TRINA_DATA_PATH = "./data/train_data_path.txt"
TEST_DATA_PATH = "./data/test_data_path.txt"
TRAIN_DATA = "./data/train_data.txt"
TEST_DATA = "./data/test_data.txt"
INVERT_INDEX_DATA = "./data/invert_index_data.txt"
KMEANS_PATH = './model/k-means.m'
CENTERS_PATH = "./model/centers.txt"

NUM_CLASSES = 8
WORD_CLASSES = 128
HEIGHT, WIDTH = 512, 765
seed(42)


def train_test_data_split(img_path, train_data_path, test_data_path):
    train_data = open(train_data_path, 'w')
    test_data = open(test_data_path, 'w')

    file_dir_list = os.listdir(img_path)
    for index1 in range(len(file_dir_list)):
        images = glob.glob(img_path+file_dir_list[index1]+'/*')
        test = randint(0, len(images))
        for index2 in range(len(images)):
            if index2 != test:
                train_data.write('{}\t{}\t{}\n'.format(images[index2], file_dir_list[index1], index1))
            else:
                test_data.write('{}\t{}\t{}\n'.format(images[index2], file_dir_list[index1], index1))
    train_data.close()
    test_data.close()


def get_all_feature_points(data_path):
    data = open(data_path, 'r')
    paths = []
    y = []
    for line in data.readlines():
        line_splited = line.strip().split('\t')
        paths.append(line_splited[0])
        y.append(line_splited[-1])
    sift = cv2.xfeatures2d.SIFT_create()
    all_features = list()
    print("Step 1: extract SIFT features from VGG’s image dataset")
    for fname in paths:
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        key_points, description = sift.detectAndCompute(gray, None)
        # display
        # img = cv2.drawKeypoints(img, key_points, img, color=(255, 0, 0))
        # cv2.imshow("sift", img)
        # cv2.waitKey(0)
        all_features.append(description)
    return all_features, paths, y


def clusting_model(all_features):
    feature_points = None
    for features in all_features:
        if feature_points is None:
            feature_points = features
        else:
            feature_points = np.concatenate([feature_points, features])
    print("Step 2: K-Means clustering on the extracted data")
    kmeans = KMeans(n_clusters=WORD_CLASSES, verbose=1, n_jobs=-1)
    kmeans.fit(feature_points)
    joblib.dump(kmeans, KMEANS_PATH)
    np.savetxt(CENTERS_PATH, kmeans.cluster_centers_, fmt="%f", delimiter=",")
    return kmeans


def calculate_the_inverted_file_index(all_features, paths, y, data_path, inverted_index_path=None):
    model = joblib.load(KMEANS_PATH)
    data = open(data_path, 'w')
    if inverted_index_path is not None:
        inverted_index_data = open(inverted_index_path, 'w')
        inverted_indexs = [[] for _ in range(WORD_CLASSES)]
    else:
        inverted_index_data = None
        inverted_indexs = None
    print("Step 3: Nearest neighbor search to get the visual words representation")
    for index in range(len(all_features)):
        word_representation = []
        for feature in all_features[index]:
            word_representation.append(model.predict(feature.reshape(1, -1))[0])
        word_representation = np.array(word_representation)
        lst = np.bincount(word_representation)
        # 防止vocabulary size 小于 WORD_CLASSES
        while len(lst) < WORD_CLASSES:
            lst = np.append(lst, 0)
        # plt.bar(range(128), lst)
        # plt.show()
        # cv2.waitKey(0)
        for i in lst:
            data.write(str(i)+",")
        data.write(str(paths[index])+","+str(y[index]))
        data.write("\n")

        # 文档倒排索引
        if inverted_index_data is not None:
            for i in range(len(lst)):
                if lst[i] != 0:
                    inverted_indexs[i].append(index)
    if inverted_index_data is not None:
        print("Step 4: Calculate the inverted file index")
        for inverted_index in inverted_indexs:
            for value in inverted_index:
                inverted_index_data.write(str(value)+",")
            inverted_index_data.write("\n")
        inverted_index_data.close()
    data.close()


def prepcoress():
    # 划分数据集
    train_test_data_split(IMG_PATH, TRINA_DATA_PATH, TEST_DATA_PATH)
    train_all_features, train_paths, y = get_all_feature_points(TRINA_DATA_PATH)
    # clusting_model(train_all_features)
    # Bag of Word
    calculate_the_inverted_file_index(train_all_features, train_paths, y, TRAIN_DATA, inverted_index_path=INVERT_INDEX_DATA)
    test_all_features, test_paths, y = get_all_feature_points(TEST_DATA_PATH)
    calculate_the_inverted_file_index(test_all_features, test_paths, y, TEST_DATA)


def load_data(data_path):
    f = open(data_path)
    X = []
    paths = []
    y = []
    for line in f:
        line = line[:-1].split(',')
        x = []
        for s in line[:-2]:
            x.append(int(s))
        X.append(x)
        paths.append(line[-2])
        y.append(int(line[-1]))
    f.close()
    return np.array(X), np.array(paths), np.array(y)


def load_invert_index_table(data_path):
    f = open(data_path)
    invert_indexs = {}
    for index, line in enumerate(f):
        line = line[:-1].split(',')
        invert_index = []
        for s in line[:-1]:
            invert_index.append(int(s))
        invert_indexs[index] = invert_index
    return invert_indexs


def predict_l1(X_train, y_train, train_paths, _X_test, test_path, invert_indexs):
    _y = np.array([[-1, float('inf')] for _ in range(5)])
    like_X = set()
    # 加速检索
    for i in range(len(_X_test)):
        for j in invert_indexs[i]:
            like_X.add(j)
    for index in like_X:
        dis = np.sum(np.abs(X_train[index] - _X_test))
        if dis < max(_y[:, 1]):
            max_idx = np.argmax(_y[:, 1])
            _y[max_idx] = [index, dis]
    _y.sort(axis=0)
    # 测试图片
    imgs = []
    img = cv2.imread(test_path)
    img = cv2.resize(img, (150, 150))
    imgs.append(img)
    for i in range(5):
        img = cv2.imread(train_paths[int(_y[:, 0][i])])
        img = cv2.resize(img, (150, 150))
        imgs.append(img)
    cv2.imshow('', np.hstack(imgs))
    cv2.waitKey()
    y_pred = []
    for i in _y[:, 0]:
        y_pred.append(y_train[int(i)])
    return y_pred


def predict_l2(X_train, y_train, train_paths, _X_test, test_path):
    _y = np.array([[-1, float('inf')] for _ in range(5)])
    for index in range(len(X_train)):
        dis = np.sum(np.square(X_train[index] - _X_test))
        if dis < max(_y[:, 1]):
            max_idx = np.argmax(_y[:, 1])
            _y[max_idx] = [index, dis]
    _y.sort(axis=0)
    # 测试图片
    imgs = []
    img = cv2.imread(test_path)
    img = cv2.resize(img, (150, 150))
    imgs.append(img)
    for i in range(5):
        img = cv2.imread(train_paths[int(_y[:, 0][i])])
        img = cv2.resize(img, (150, 150))
        imgs.append(img)
    cv2.imshow('', np.hstack(imgs))
    cv2.waitKey()
    y_pred = []
    for i in _y[:, 0]:
        y_pred.append(y_train[int(i)])
    return y_pred


def get_acc(y, y_hat):
    y_hat = np.array(y_hat)
    print("top1 accuracy:", sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat[:, 0])) / len(y))
    print("top5 accuracy:", sum(sum(yi == yi_hat for yi, yi_hat in zip(y, y_hat))) / len(y) / 5)


def get_precision(y, y_hat):
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            confusion_matrix[i][j] = sum(yi == i and yi_hat[0] == j for yi, yi_hat in zip(y, y_hat))
    _sum = confusion_matrix.sum(axis=0)
    for i in range(len(confusion_matrix)):
        if _sum[i] != 0:
            print("class ", i, "top1 precision:", confusion_matrix[i][i] / _sum[i] * 100, "%")
        else:
            print("class ", i, "top1 precision:", 0, "%")
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            for k in range(5):
                confusion_matrix[i][j] += sum(yi == i and yi_hat[k] == j for yi, yi_hat in zip(y, y_hat))
    _sum = confusion_matrix.sum(axis=0)
    for i in range(len(confusion_matrix)):
        if _sum[i] != 0:
            print("class ", i, "top5 precision:", confusion_matrix[i][i] / _sum[i] * 100, "%")
        else:
            print("class ", i, "top5 precision:", 0, "%")


def get_recall(y, y_hat):
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            confusion_matrix[i][j] = sum(yi == i and yi_hat[0] == j for yi, yi_hat in zip(y, y_hat))
    _sum = confusion_matrix.sum(axis=1)
    for i in range(len(confusion_matrix)):
        print("class ", i, " recall:", confusion_matrix[i][i] / _sum[i] * 100, "%")
    confusion_matrix = np.zeros((NUM_CLASSES, NUM_CLASSES))
    for i in range(NUM_CLASSES):
        for j in range(NUM_CLASSES):
            for k in range(5):
                confusion_matrix[i][j] += sum(yi == i and yi_hat[k] == j for yi, yi_hat in zip(y, y_hat))
    _sum = confusion_matrix.sum(axis=1)
    for i in range(len(confusion_matrix)):
        if _sum[i] != 0:
            print("class ", i, "top5 precision:", confusion_matrix[i][i] / _sum[i] * 100, "%")
        else:
            print("class ", i, "top5 precision:", 0, "%")


def evalutaion(X_train, y_train, train_paths, X_test, y_test, test_paths, invert_indexs):
    y_pred = []
    for i in range(len(X_test)):
        y_pred.append(predict_l1(X_train, y_train, train_paths, X_test[i], test_paths[i], invert_indexs))
    get_acc(y_test, y_pred)
    get_precision(y_test, y_pred)
    get_recall(y_test, y_pred)


def main():
    # prepcoress()
    print("Step 5: Evaluation")
    X_train, train_paths, y_train = load_data(TRAIN_DATA)
    X_test, test_paths, y_test = load_data(TEST_DATA)
    invert_indexs = load_invert_index_table(INVERT_INDEX_DATA)
    evalutaion(X_train, y_train, train_paths,  X_test, y_test, test_paths, invert_indexs)


if __name__ == '__main__':
    main()
    print()

