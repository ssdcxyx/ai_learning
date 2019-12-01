# -*- coding: utf-8 -*-
# @Time    : 2019-05-17 20:24
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Model.py

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from handle import DataProcess


def kmeans(data):
    data = data.drop(['user_id'], axis=1)
    model = PCA(n_components=2)
    data = model.fit_transform(data)
    data = DataProcess.number_normal(data)
    model = KMeans(n_clusters=4)
    model.fit(data)
    return model.labels_


def tsne(data):
    model = TSNE(n_components=2, learning_rate=200)
    data = model.fit_transform(data)
    return data


def pca(data):
    model = PCA(n_components=2)
    data = model.fit_transform(data)
    return data


if __name__ == '__main__':
    print()


