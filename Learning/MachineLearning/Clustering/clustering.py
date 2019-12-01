# -*- coding: utf-8 -*-
# @Time    : 2018-12-17 19:01
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : clustering.py

import numpy as np
import os
from sklearn.datasets import load_iris
from sklearn.mixture import GaussianMixture
from sklearn.datasets import make_blobs
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from timeit import timeit
from sklearn.metrics import silhouette_score
from sklearn.metrics import silhouette_samples
from matplotlib.ticker import FixedLocator, FixedFormatter
from matplotlib.image import imread
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import make_moons
from sklearn.cluster import DBSCAN
from sklearn.neighbors import KNeighborsClassifier
from sklearn.cluster import SpectralClustering
from sklearn.cluster import AgglomerativeClustering
from matplotlib.colors import LogNorm
from sklearn.mixture import BayesianGaussianMixture
from scipy.stats import norm
from matplotlib.patches import Polygon

np.random.seed(42)

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "clustering"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# img = load_iris()
# X = img.img
# y = img.target
# print(img.target_names)

# plt.figure(figsize=(9, 3.5))
#
# plt.subplot(121)
# plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")
# plt.plot(X[y == 1, 0], X[y == 1, 1], 'bs', label="Iris-Versicolor")
# plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
# plt.xlabel("Petal width", fontsize=14)
# plt.ylabel("Petal length", fontsize=14)
# plt.legend(fontsize=12)
#
# plt.subplot(122)
# plt.scatter(X[:, 0], X[:, 1], c="k", marker=".")
# plt.xlabel("Petal length", fontsize=14)
# plt.tick_params(labelleft='off')
#
# save_fig("classification_vs_clustering_diagram")
# plt.show()

# 高斯混合模型
# y_pred = GaussianMixture(n_components=3, random_state=42).fit(X).predict(X)
# mapping = np.array([2, 0, 1])
# y_pred = np.array([mapping[cluster_id] for cluster_id in y_pred])
#
# plt.plot(X[y_pred == 0, 0], X[y_pred == 0, 1], "yo", label="Cluster 1")
# plt.plot(X[y_pred == 1, 0], X[y_pred == 1, 1], "bs", label="Cluster 2")
# plt.plot(X[y_pred == 2, 0], X[y_pred == 2, 1], "g^", label="Cluster 3")
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(loc="upper right", fontsize=12)
# plt.show()
#
# print(np.sum(y_pred == y) / len(y_pred))

blob_centers = np.array(
    [[0.2, 2.3],
     [-1.5, 2.3],
     [-2.8, 1.8],
     [-2.8, 2.8],
     [-2.8, 1.3]]
)
blob_std = np.array([0.4, 0.3, 0.1, 0.1, 0.1])

X, y = make_blobs(n_samples=2000, centers=blob_centers,
                  cluster_std=blob_std, random_state=7)


def plot_clusters(X, y=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=1)
    plt.xlabel("$x_1$", fontsize=14)
    plt.ylabel("$y_1$", fontsize=14, rotation=0)


# plt.figure(figsize=(8, 4))
# plot_clusters(X)
# save_fig("blobs_diagram")
# plt.show()

# K-Means聚类
k = 5
kmeans = KMeans(n_clusters=k, random_state=42)
y_pred = kmeans.fit_predict(X)

# print(y_pred)
#
# print(y_pred is kmeans.labels_)
#
# print(kmeans.cluster_centers_)
#
# print(kmeans.labels_)
#
X_new = np.array([[0, 2], [3, 2], [-3, 3], [-3, 2.5]])
# print(kmeans.predict(X_new))


def plot_data(X):
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)


def plot_centroids(centroids, weights=None, circle_color='w', cross_color='k'):
    if weights is not None:
        centroids = centroids[weights > weights.max() / 10]
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='o', s=30, linewidths=8,
                color=circle_color, zorder=10, alpha=0.9)
    plt.scatter(centroids[:, 0], centroids[:, 1],
                marker='x', s=50, linewidths=50,
                color=cross_color, zorder=11, alpha=1)


def plot_decision_boundaries(clusterer, X, resolution=1000, show_centroids=True,
                             show_xlabels=True, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), cmap="Pastel2")
    plt.contour(Z, extent=(mins[0], maxs[0], mins[1], maxs[1]), linewidth=1, color='k')
    plot_data(X)
    if show_centroids:
        plot_centroids(clusterer.cluster_centers_)
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')


# plt.figure(figsize=(8, 4))
# plot_decision_boundaries(kmeans, X)
# save_fig("voronnoi_diagram")
# plt.show()

# print(kmeans.transform(X_new))

# print(np.linalg.norm(np.tile(X_new, (1, k)).reshape(-1, k, 2) - kmeans.cluster_centers_, axis=2))

# kmeans_iter1 = KMeans(n_clusters=5, init="random", n_init=1,
#                       algorithm="full", max_iter=1, random_state=1)
# kmeans_iter2 = KMeans(n_clusters=5, init="random", n_init=1,
#                       algorithm="full", max_iter=2, random_state=1)
# kmeans_iter3 = KMeans(n_clusters=5, init="random", n_init=1,
#                       algorithm="full", max_iter=3, random_state=1)
# kmeans_iter1.fit(X)
# kmeans_iter2.fit(X)
# kmeans_iter3.fit(X)
#
# plt.figure(figsize=(10, 8))
#
# plt.subplot(321)
# plot_data(X)
# plot_centroids(kmeans_iter1.cluster_centers_, circle_color='r',
#                cross_color='w')
# plt.ylabel("$x_2$", fontsize=14, rotation=0)
# plt.tick_params(labelbottom='off')
# plt.title("Update the centroids (initially randomly)", fontsize=14)
#
# plt.subplot(322)
# plot_decision_boundaries(kmeans_iter1, X, show_xlabels=False, show_ylabels=False)
# plt.title("Label the instances", fontsize=14)
#
# plt.subplot(323)
# plot_decision_boundaries(kmeans_iter1, X, show_centroids=False, show_xlabels=False)
# plot_centroids(kmeans_iter2.cluster_centers_)
#
# plt.subplot(324)
# plot_decision_boundaries(kmeans_iter2, X, show_xlabels=False, show_ylabels=False)
#
# plt.subplot(325)
# plot_decision_boundaries(kmeans_iter2, X, show_centroids=False, show_xlabels=False)
# plot_centroids(kmeans_iter3.cluster_centers_)
#
# plt.subplot(326)
# plot_decision_boundaries(kmeans_iter3, X, show_xlabels=False, show_ylabels=False)
#
# save_fig("kmeans_algorith_diagram")
# plt.show()


def plot_clusterer_comparison(clusterer1, clusterer2, X, title1=None, title2=None):
    clusterer1.fit(X)
    clusterer2.fit(X)

    plt.figure(figsize=(10, 3.2))

    plt.subplot(121)
    plot_decision_boundaries(clusterer1, X)
    if title1:
        plt.title(title1, fontsize=14)

        plt.subplot(122)
        plot_decision_boundaries(clusterer2, X, show_ylabels=False)
        if title2:
            plt.title(title2, fontsize=14)


# kmeans_rnd_init1 = KMeans(n_clusters=5, init="random", n_init=1,
#                           algorithm="full", random_state=11)
# kmeans_rnd_init2 = KMeans(n_clusters=5, init="random", n_init=1,
#                           algorithm="full", random_state=19)
#
# plot_clusterer_comparison(kmeans_rnd_init1, kmeans_rnd_init2, X,
#                           "Solution 1", "Solution 2 (with a different random init)")
# save_fig("kmeans_variability_diagram")
# plt.show()

# print(kmeans.inertia_)
#
# X_dist = kmeans.transform(X)
# print(np.sum(X_dist[np.arange(len(X_dist)), kmeans.labels_]**2))
#
# print(kmeans.score(X))

# print(kmeans_rnd_init1.inertia_)
#
# print(kmeans_rnd_init2.inertia_)

# kmeans_rnd_10_inits = KMeans(n_clusters=5, init="random", n_init=10,
#                              algorithm="full", random_state=11)
# print(kmeans_rnd_10_inits.fit(X))
#
# plt.figure(figsize=(8, 4))
# plot_decision_boundaries(kmeans_rnd_10_inits, X)
# plt.show()

# K-Means++
# good_init = np.array([[-3, 3], [-3, 2], [-3, 1], [-1, 2], [0, 2]])
# kmeans = KMeans(n_clusters=5, init=good_init, n_init=1, random_state=42)
# kmeans.fit(X)
# print(kmeans.inertia_)

# 加速 K-Means(elkan)
# print(KMeans(algorithm="elkan").fit(X))

# Mini-batch K-Means
# minibatch_kmeans = MiniBatchKMeans(n_clusters=5, random_state=42)
# print(minibatch_kmeans.fit(X))
#
# print(minibatch_kmeans.inertia_)

# filename = "my_mnist.img"
# m, n = 50000, 28*28
# X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
#
# minibatch_kmeans = MiniBatchKMeans(n_clusters=10, batch_size=10, random_state=42)
# print(minibatch_kmeans.fit(X_mm))


def load_next_batch(batch_size):
    return X[np.random.choice(len(X), batch_size, replace=False)]


# np.random.seed(42)
#
# k = 5
# n_init = 10
# n_iterations = 100
# batch_size = 100
# init_size = 500
# evaluate_on_last_n_iters = 10
#
# best_kmeans = None
#
# for init in range(n_init):
#     minibatch_kmeans = MiniBatchKMeans(n_clusters=k, init_size=init_size)
#     X_init = load_next_batch(init_size)
#     minibatch_kmeans.partial_fit(X_init)
#
#     minibatch_kmeans.sum_inertia_ = 0
#     for iteration in range(n_iterations):
#         X_batch = load_next_batch(batch_size)
#         minibatch_kmeans.partial_fit(X_batch)
#         if iteration >= n_iterations - evaluate_on_last_n_iters:
#             minibatch_kmeans.sum_inertia_ += minibatch_kmeans.inertia_
#
#     if best_kmeans is None or minibatch_kmeans.sum_inertia_ < best_kmeans.sum_inertia_:
#         best_kmeans = minibatch_kmeans
#
# print(best_kmeans.score(X))

# times = np.empty((100, 2))
# inertias = np.empty((100, 2))
# for k in range(1, 101):
#     kmeans = KMeans(n_clusters=k, random_state=42)
#     minibatch_kmeans = MiniBatchKMeans(n_clusters=k, random_state=42)
#     print("\r{}/{}".format(k, 100), end="")
#     times[k-1, 0] = timeit("kmeans.fit(X)", number=10, globals=globals())
#     times[k-1, 1] = timeit("minibatch_kmeans.fit(X)", number=10, globals=globals())
#     inertias[k-1, 0] = kmeans.inertia_
#     inertias[k-1, 1] = minibatch_kmeans.inertia_
#
#
# plt.figure(figsize=(10, 4))
#
# plt.subplot(121)
# plt.plot(range(1, 101), inertias[:, 0], "r--", label="K-Means")
# plt.plot(range(1, 101), inertias[:, 1], "b.-", label="Mini-batch K-Means")
# plt.xlabel("$k$", fontsize=16)
# plt.title("Inertia", fontsize=14)
# plt.legend(fontsize=14)
# plt.axis([1, 100, 0, 100])
#
# plt.subplot(122)
# plt.plot(range(1, 101), times[:, 0], "r--", label="K-Means")
# plt.plot(range(1, 101), times[:, 1], "b.-", label="Mini-batch K-Means")
# plt.xlabel("$k$", fontsize=16)
# plt.title("Traing time (seconds)", fontsize=14)
# plt.axis([1, 100, 0, 6])
#
# save_fig("minibatch_kmeans_vs_kmeans")
# plt.show()

# kmeans_k3 = KMeans(n_clusters=3, random_state=42)
# kmeans_k8 = KMeans(n_clusters=8, random_state=42)
#
# plot_clusterer_comparison(kmeans_k3, kmeans_k8, X, "$k=3$", "$k=8$")
# save_fig("bad_n_clusters_diagram")
# plt.show()
#
# print(kmeans_k3.inertia_)
#
# print(kmeans_k8.inertia_)

kmeans_per_k = [KMeans(n_clusters=k, random_state=42).fit(X)
                for k in range(1, 10)]
# inertias = [model.inertia_ for model in kmeans_per_k]

# plt.figure(figsize=(8, 3.5))
# plt.plot(range(1, 10), inertias, "bo-")
# plt.xlabel("$k$", fontsize=14)
# plt.ylabel("Inertia", fontsize=14)
# plt.annotate('Elbow',
#              xy=(4, inertias[3]),
#              xytext=(0.55, 0.55),
#              textcoords='figure fraction',
#              fontsize=16,
#              arrowprops=dict(facecolor='black', shrink=0.1))
# plt.axis([1, 8.5, 0, 1300])
# save_fig("inertia_vs_k_diagram")
# plt.show()

# plot_decision_boundaries(kmeans_per_k[4-1], X)
# plt.show()

# print(silhouette_score(X, kmeans.labels_))

silhouette_scores = [silhouette_score(X, model.labels_)
                     for model in kmeans_per_k[1:]]
# plt.figure(figsize=(8, 3))
# plt.plot(range(2, 10), silhouette_scores, "bo-")
# plt.xlabel("$k$", fontsize=14)
# plt.ylabel("Silhouette score", fontsize=14)
# plt.axis([1.8, 8.5, 0.55, 0.7])
# save_fig("silhouette_score_vs_k_diagram")
# plt.show()

# plt.figure(figsize=(11, 9))
#
# for k in (3, 4, 5, 6):
#     plt.subplot(2, 2, k - 2)
#     y_pred = kmeans_per_k[k - 1].labels_
#     silhouette_coefficients = silhouette_samples(X, y_pred)
#
#     padding = len(X) // 30
#     pos = padding
#     ticks = []
#     for i in range(k):
#         coeffs = silhouette_coefficients[y_pred == i]
#         coeffs.sort()
#
#         color = matplotlib.cm.Spectral(i / k)
#         plt.fill_betweenx(np.arange(pos, pos + len(coeffs)), 0, coeffs,
#                           facecolor=color, edgecolor=color, alpha=0.7)
#         ticks.append(pos + len(coeffs) // 2)
#         pos += len(coeffs) + padding
#
#     plt.gca().yaxis.set_major_locator(FixedLocator(ticks))
#     plt.gca().yaxis.set_major_formatter(FixedFormatter(range(k)))
#     if k in (3, 5):
#         plt.ylabel("Cluster")
#
#     if k in (5, 6):
#         plt.gca().set_xticks([-0.1, 0, 0.2, 0.4, 0.6, 0.8, 1])
#         plt.xlabel("Silhouetto Coefficient")
#     else:
#         plt.tick_params(labelbottom='off')
#     plt.axvline(x=silhouette_scores[k-2], color="red", linestyle="--")
#     plt.title("$k={}$".format(k), fontsize=16)
#
# save_fig("silhouette_analysis_diagram")
# plt.show()

# X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
# X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
# X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
# X2 = X2 + [6, -8]
# X = np.r_[X1, X2]
# y = np.r_[y1, y2]
#
# plot_clusters(X)
# plt.show()

# kmeans_good = KMeans(n_clusters=3, init=np.array([[-1.5, 2.5], [0.5, 0], [4, 0]]), n_init=1, random_state=42)
# kmeans_bad = KMeans(n_clusters=3, random_state=42)
# kmeans_good.fit(X)
# print(kmeans_bad.fit(X))
#
# plt.figure(figsize=(10, 3.2))
# plt.subplot(121)
# plot_decision_boundaries(kmeans_good, X)
# plt.title("Inertia = {:.1f}".format(kmeans_good.inertia_), fontsize=14)
#
# plt.subplot(122)
# plot_decision_boundaries(kmeans_bad, X, show_ylabels=False)
# plt.title("Inertia = {:.1f}".format(kmeans_bad.inertia_), fontsize=14)
#
# save_fig("bad_kmeans_diagram")
# plt.show()

# image = imread(os.path.join("images", "", "ladybug.png"))
# print(image.shape)
#
# X = image.reshape(-1, 3)
# kmeans = KMeans(n_clusters=8, random_state=42).fit(X)
# segmented_img = kmeans.cluster_centers_[kmeans.labels_]
# segmented_img = segmented_img.reshape(image.shape)
#
# segmented_imgs = []
# n_colors = (10, 8, 6, 4, 2)
# for n_clusters in n_colors:
#     kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(X)
#     segmented_img = kmeans.cluster_centers_[kmeans.labels_]
#     segmented_imgs.append(segmented_img.reshape(image.shape))
#
# plt.figure(figsize=(10, 5))
# plt.subplots_adjust(wspace=0.05, hspace=0.1)
#
# plt.subplot(231)
# plt.imshow(image)
# plt.title("Original image")
# plt.axis("off")
#
# for idx, n_clusters in enumerate(n_colors):
#     plt.subplot(232 + idx)
#     plt.imshow(segmented_imgs[idx])
#     plt.title("{} colors".format(n_clusters))
#     plt.axis('off')
#
# save_fig('image_segmentation_diagram', tight_layout=False)
# plt.show()

X_digits, y_digits = load_digits(return_X_y=True)

X_train, X_test, y_train, y_test = train_test_split(X_digits, y_digits, random_state=42)

log_reg = LogisticRegression(random_state=42)
# print(log_reg.fit(X_train, y_train))
#
# print(log_reg.score(X_test, y_test))

pipeline = Pipeline([
    ("kmeans", KMeans(n_clusters=50, random_state=42)),
    ("log_reg", LogisticRegression(random_state=42))
])
# print(pipeline.fit(X_train, y_train))
#
# print(pipeline.score(X_test, y_test))

# param_grid = dict(kmeans__n_clusters=range(2, 100))
# grid_clf = GridSearchCV(pipeline, param_grid, cv=3, verbose=2)
# grid_clf.fit(X_train, y_train)
#
#
# print(grid_clf.best_params_)
#
# print(grid_clf.score(X_test, y_test))

# n_labeled = 50
# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(X_train[:n_labeled], y_train[:n_labeled])
# print(log_reg.score(X_test, y_test))

# k = 50
#
# kmeans = KMeans(n_clusters=k, random_state=42)
# X_digits_dist = kmeans.fit_transform(X_train)
# representative_digit_idx = np.argmin(X_digits_dist, axis=0)
# X_representative_digits = X_train[representative_digit_idx]
#
# plt.figure(figsize=(8, 2))
# for index, X_representative_digit in enumerate(X_representative_digits):
#     plt.subplot(k // 10, 10, index+1)
#     plt.imshow(X_representative_digit.reshape(8, 8), cmap="binary", interpolation="bilinear")
#     plt.axis("off")
#
# save_fig("representative_images_diagram", tight_layout=False)
# plt.show()

# y_representative_digits = np.array([
#     4, 8, 0, 6, 7, 3, 7, 7, 9, 2,
#     5, 5, 8, 5, 2, 1, 2, 9, 6, 1,
#     1, 6, 9, 0, 8, 3, 0, 7, 4, 1,
#     6, 5, 2, 4, 1, 8, 6, 3, 9, 2,
#     4, 2, 9, 4, 7, 6, 2, 3, 1, 1
# ])
#
# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(X_representative_digits, y_representative_digits)
# print(log_reg.score(X_test, y_test))

# y_train_propagated = np.empty(len(X_train), dtype=np.int32)
# for i in range(k):
#     y_train_propagated[kmeans.labels_ == i] = y_representative_digits[i]

# log_reg = LogisticRegression(random_state=42)
# print(log_reg.fit(X_train, y_train_propagated))
#
# print(log_reg.score(X_test, y_test))

# percentile_closest = 20
#
# X_cluster_dist = X_digits_dist[np.arange(len(X_train)), kmeans.labels_]
# for i in range(k):
#     in_cluster = (kmeans.labels_ == i)
#     cluster_dist = X_cluster_dist[in_cluster]
#     cutoff_distance = np.percentile(cluster_dist, percentile_closest)
#     above_cutoff = (X_cluster_dist > cutoff_distance)
#     X_cluster_dist[in_cluster & above_cutoff] = -1
#
# partially_propagated = (X_cluster_dist != -1)
# X_train_partially_propagated = X_train[partially_propagated]
# y_train_partially_propagated = y_train_propagated[partially_propagated]
#
# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(X_train_partially_propagated, y_train_partially_propagated)
#
# print(log_reg.score(X_test, y_test))
#
# print(np.mean(y_train_partially_propagated == y_train[partially_propagated]))

X, y = make_moons(n_samples=1000, noise=0.05, random_state=42)

# DBSCAN
# dbscan = DBSCAN(eps=0.05, min_samples=5)
# dbscan.fit(X)

# print(dbscan.labels_[:10])
#
# print(len(dbscan.core_sample_indices_))
#
# print(dbscan.components_[:3])
#
# print(np.unique(dbscan.labels_))

# dbscan2 = DBSCAN(eps=0.2)
# dbscan2.fit(X)


def plot_dbscan(dbscan, X, size, show_xlabels=True, show_ylabels=True):
    core_mask = np.zeros_like(dbscan.labels_, dtype=bool)
    core_mask[dbscan.core_sample_indices_] = True
    anomalies_mask = dbscan.labels_ == -1
    non_core_mask = ~(core_mask | anomalies_mask)

    cores = dbscan.components_
    anomalies = X[anomalies_mask]
    non_cores = X[non_core_mask]

    plt.scatter(cores[:, 0], cores[:, 1],
                c=dbscan.labels_[core_mask], marker='o', s=size, cmap="Paired")
    plt.scatter(cores[:, 0], cores[:, 1], marker='*', s=20, c=dbscan.labels_[core_mask])
    plt.scatter(anomalies[:, 0], anomalies[:, 1], c="r", marker="x", s=100)
    plt.scatter(non_cores[:, 0], non_cores[:, 1], c=dbscan.labels_[non_core_mask], marker=".")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')
    plt.title("eps={:.2f}, min_samples={}".format(dbscan.eps, dbscan.min_samples), fontsize=14)


# plt.figure(figsize=(9, 3.2))
# plt.subplot(121)
# plot_dbscan(dbscan, X, size=100)
#
# plt.subplot(122)
# plot_dbscan(dbscan2, X, size=600, show_ylabels=False)
#
# save_fig("dbscan_diagram")
# plt.show()

# dbscan = dbscan2
#
# knn = KNeighborsClassifier(n_neighbors=50)
# print(knn.fit(dbscan.components_, dbscan.labels_[dbscan.core_sample_indices_]))
#
# X_new = np.array([[-0.5, 0], [0, 0.5], [1, -0.1], [2, 1]])
# print(knn.predict(X_new))
#
# print(knn.predict_proba(X_new))
#
# plt.figure(figsize=(6, 3))
# plot_decision_boundaries(knn, X, show_centroids=False)
# plt.scatter(X_new[:, 0], X_new[:, 1], c="b", marker="+", s=200, zorder=10)
# save_fig("cluster_classification_diagram")
# plt.show()
#
# y_dist, y_pred_idx = knn.kneighbors(X_new, n_neighbors=1)
# y_pred = dbscan.labels_[dbscan.core_sample_indices_][y_pred_idx]
# y_pred[y_dist > 0.2] = -1
# print(y_pred.ravel())

# Spectral Culstering （谱聚类）
# sc1 = SpectralClustering(n_clusters=2, gamma=100, random_state=42)
# print(sc1.fit(X))
#
# sc2 = SpectralClustering(n_clusters=2, gamma=1, random_state=42)
# print(sc2.fit(X))


def plot_spectral_clustering(sc, X, size, alpha, show_xlabels=True, show_ylabels=True):
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=size, c='gray', cmap='Paired', alpha=alpha)
    plt.scatter(X[:, 0], X[:, 1], marker='o', s=30, c='w')
    plt.scatter(X[:, 0], X[:, 1], marker='.', s=10, c=sc.labels_, cmap="Paired")
    if show_xlabels:
        plt.xlabel("$x_1$", fontsize=14)
    else:
        plt.tick_params(labelbottom='off')
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft="off")
    plt.title("RBF gamma={}".format(sc.gamma), fontsize=14)


# plt.figure(figsize=(9, 3.2))
# plt.subplot(121)
# plot_spectral_clustering(sc1, X, size=500, alpha=0.1)
#
# plt.subplot(122)
# plot_spectral_clustering(sc2, X, size=4000, alpha=0.01, show_ylabels=False)
# plt.show()


def learned_parameters(model):
    return [m for m in dir(model)
            if m.endswith("_") and not m.startswith("_")]


# Agglomerative Hierarchical Clustering(自底向上的层次聚类）
# X = np.array([0, 2, 5, 8.5]).reshape(-1, 1)
# agg = AgglomerativeClustering(linkage="complete").fit(X)
#
# print(learned_parameters(agg))
#
# print(agg.children_)

X1, y1 = make_blobs(n_samples=1000, centers=((4, -4), (0, 0)), random_state=42)
X1 = X1.dot(np.array([[0.374, 0.95], [0.732, 0.598]]))
X2, y2 = make_blobs(n_samples=250, centers=1, random_state=42)
X2 = X2 + [6, -8]
X = np.r_[X1, X2]
y = np.r_[y1, y2]

gm = GaussianMixture(n_components=3, n_init=3, random_state=42)
gm.fit(X)

# print(gm.weights_)
#
# print(gm.means_)
#
# print(gm.covariances_)
#
# print(gm.converged_)
#
# print(gm.n_iter_)
#
# print(gm.predict(X))
#
# print(gm.predict_proba(X))
#
# X_new, y_new = gm.sample(6)
#
# print(X_new)
#
# print(y_new)
#
# print(gm.score_samples(X))

resolution = 100
grid = np.arange(-10, 10, 1 / resolution)
xx, yy = np.meshgrid(grid, grid)
X_full = np.vstack([xx.ravel(), yy.ravel()]).T

pdf = np.exp(gm.score_samples(X_full))
pdf_probas = pdf * (1/resolution) ** 2
# print(pdf_probas.sum())


def plot_gaussian_mixture(clusterer, X, resolution=1000, show_ylabels=True):
    mins = X.min(axis=0) - 0.1
    maxs = X.max(axis=0) + 0.1
    xx, yy = np.meshgrid(np.linspace(mins[0], maxs[0], resolution),
                         np.linspace(mins[1], maxs[1], resolution))
    Z = -clusterer.score_samples(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contourf(xx, yy, Z,
                 norm=LogNorm(vmin=1.0, vmax=30.0),
                 levels=np.logspace(0, 2, 12))
    plt.contour(xx, yy, Z,
                norm=LogNorm(vmin=1.0, vmax=30.0),
                levels=np.logspace(0, 2, 12),
                linewidths=1, colors='k')
    Z = clusterer.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    plt.contour(xx, yy, Z,
                linewidth=2, color='r', linestyles="dashed")
    plt.plot(X[:, 0], X[:, 1], 'k.', markersize=2)
    plot_centroids(clusterer.means_, clusterer.weights_)
    plt.xlabel("$x_1$", fontsize=14)
    if show_ylabels:
        plt.ylabel("$x_2$", fontsize=14, rotation=0)
    else:
        plt.tick_params(labelleft='off')


# plt.figure(figsize=(8, 4))
# plot_gaussian_mixture(gm, X)
# save_fig("gaussian_mixture_diagram")
# plt.show()

# gm_full = GaussianMixture(n_components=3, n_init=10, covariance_type="full", random_state=42)
# gm_tied = GaussianMixture(n_components=3, n_init=10, covariance_type="tied", random_state=42)
# gm_spherical = GaussianMixture(n_components=3, n_init=10, covariance_type="spherical", random_state=42)
# gm_diag = GaussianMixture(n_components=3, n_init=10, covariance_type="diag", random_state=42)
# gm_full.fit(X)
# gm_tied.fit(X)
# gm_spherical.fit(X)
# gm_diag.fit(X)


def compare_gaussian_mixtures(gm1, gm2, X):
    plt.figure(figsize=(9, 4))

    plt.subplot(121)
    plot_gaussian_mixture(gm1, X)
    plt.title('covariance_type="{}"'.format(gm1.covariance_type), fontsize=14)

    plt.subplot(122)
    plot_gaussian_mixture(gm2, X, show_ylabels=False)
    plt.title('covariane_type='"{}".format(gm2.covariance_type), fontsize=14)


# compare_gaussian_mixtures(gm_tied, gm_spherical, X)
# save_fig("convariance_type_diagram")
# plt.show()

# compare_gaussian_mixtures(gm_full, gm_diag, X)
# plt.tight_layout()
# plt.show()

# densities = gm.score_samples(X)
# density_threshold = np.percentile(densities, 4)
# anomalies = X[densities < density_threshold]
#
# plt.figure(figsize=(8, 4))
#
# plot_gaussian_mixture(gm, X)
# plt.scatter(anomalies[:, 0], anomalies[:, 1], color='r', marker='*')
# plt.ylim(ymax=5.1)
#
# save_fig("mixture_anomaly_detection_diagram")
# plt.show()

# BIC 贝叶斯信息准则 AIC Akaike信息准则
# print(gm.bic(X))
#
# print(gm.aic(X))

# n_cluster = 3
# n_dims = 2
# n_params_for_weights = n_cluster - 1
# n_params_for_means = n_cluster * n_dims
# n_params_for_covariance = n_cluster * n_dims * (n_dims + 1) // 2
# n_params = n_params_for_weights + n_params_for_means + n_params_for_covariance
# max_log_likehood = gm.score(X) * len(X)
# bic = np.log(len(X)) * n_params -2 * max_log_likehood
# aic = 2 * n_params - 2 * max_log_likehood
# print(bic, aic)
#
# print(n_params)

# gms_per_k = [GaussianMixture(n_components=k, n_init=10, random_state=42).fit(X)
#              for k in range(1, 11)]
#
# bics = [model.bic(X) for model in gms_per_k]
# aics = [model.aic(X) for model in gms_per_k]

# plt.figure(figsize=(8, 3))
# plt.plot(range(1, 11), bics, "bo--", label="BIC")
# plt.plot(range(1, 11), aics, "go--", label="AIC")
# plt.xlabel("$k$", fontsize=14)
# plt.ylabel("Information Criterion", fontsize=14)
# plt.axis([1, 9.5, np.min(aics) - 50, np.max(aics) + 50])
# plt.annotate('Minimum',
#              xy=(3, bics[2]),
#              xytext=(0.35, 0.6),
#              textcoords='figure fraction',
#              fontsize=14,
#              arrowprops=dict(facecolor='black', shrink=0.1))
# plt.legend()
# save_fig("aic_bic_vs_k_diagram")
# plt.show()

# min_bic = np.infty
#
# for k in range(1, 11):
#     for covariance_type in ("full", "tied", "spherical", "diag"):
#         bic = GaussianMixture(n_components=k, n_init=10, covariance_type=covariance_type, random_state=42).fit(X).bic(X)
#         if bic < min_bic:
#             min_bic = bic
#             best_k = k
#             best_covariance_type = covariance_type
#
# print(best_k)
#
# print(best_covariance_type)

# bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
# bgm.fit(X)
#
# print(np.round(bgm.weights_, 2))
#
# plt.figure(figsize=(8, 5))
# plot_gaussian_mixture(bgm, X)
# plt.show()

# bgm_low = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
#                                   weight_concentration_prior=0.01, random_state=42)
# bgm_high = BayesianGaussianMixture(n_components=10, max_iter=1000, n_init=1,
#                                    weight_concentration_prior=10000, random_state=42)
# nn = 73
# bgm_low.fit(X[:nn])
# bgm_high.fit(X[:nn])
#
# print(np.round(bgm_low.weights_, 2))
#
# print(np.round(bgm_high.weights_, 2))
#
#
# plt.figure(figsize=(9, 4))
#
# plt.subplot(121)
# plot_gaussian_mixture(bgm_low, X[:nn])
# plt.title("weight_concentration_prior=0.01", fontsize=14)
# plt.subplot(122)
# plot_gaussian_mixture(bgm_high, X[:nn], show_ylabels=False)
# plt.title("weight_concentration_prior = 10000", fontsize=14)
#
# save_fig("mixture_concentration_prior_diagram")
# plt.show()

# X_moons, y_moons = make_moons(n_samples=1000, noise=0.05, random_state=42)
# bgm = BayesianGaussianMixture(n_components=10, n_init=10, random_state=42)
# bgm.fit(X_moons)
#
# plt.figure(figsize=(9, 3.2))
#
# plt.subplot(121)
# plot_data(X_moons)
# plt.xlabel("$x_1$", fontsize=14)
# plt.ylabel("$x_2$", fontsize=14, rotation=0)
#
# plt.subplot(122)
# plot_gaussian_mixture(bgm, X_moons, show_ylabels=False)
# save_fig("moons_vs_bgm_diagram")
# plt.show()

xx = np.linspace(-6, 4, 101)
ss = np.linspace(1, 2, 101)
XX, SS = np.meshgrid(xx, ss)
ZZ = 2 * norm.pdf(XX - 1.0, 0, SS) + norm.pdf(XX + 4.0, 0, SS)
ZZ = ZZ / ZZ.sum(axis=1) / (xx[1] - xx[0])

plt.figure(figsize=(8, 4.5))
x_idx = 85
s_idx = 30

plt.subplot(221)
plt.contourf(XX, SS, ZZ, cmap="GnBu")
plt.plot([-6, 4], [ss[s_idx], ss[s_idx]], "k-", linewidth=2)
plt.plot([xx[x_idx], xx[x_idx]], [1, 2], "b-", linewidth=2)
plt.xlabel(r"$x$")
plt.ylabel(r"$\theta$", fontsize=14, rotation=0)
plt.title(r"Model $f(x; \theta)$", fontsize=14)

plt.subplot(222)
plt.plot(ss, ZZ[:, x_idx], "b-")
max_idx = np.argmax(ZZ[:, x_idx])
max_val = np.max(ZZ[:, x_idx])
plt.plot(ss[max_idx], max_val, "r.")
plt.plot([ss[max_idx], ss[max_idx]], [0, max_val], "r:")
plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")
plt.text(1.01, max_val + 0.005, r"$\hat{L}$", fontsize=14)
plt.text(ss[max_idx]+ 0.01, 0.055, r"$\hat{\theta}$", fontsize=14)
plt.text(ss[max_idx]+ 0.01, max_val - 0.012, r"$Max$", fontsize=12)
plt.axis([1, 2, 0.05, 0.15])
plt.xlabel(r"$\theta$", fontsize=14)
plt.grid(True)
plt.text(1.99, 0.135, r"$=f(x=2.5; \theta)$", fontsize=14, ha="right")
plt.title(r"Likelihood function $\mathcal{L}(\theta|x=2.5)$", fontsize=14)

plt.subplot(223)
plt.plot(xx, ZZ[s_idx], "k-")
plt.axis([-6, 4, 0, 0.25])
plt.xlabel(r"$x$", fontsize=14)
plt.grid(True)
plt.title(r"PDF $f(x; \theta=1.3)$", fontsize=14)
verts = [(xx[41], 0)] + list(zip(xx[41:81], ZZ[s_idx, 41:81])) + [(xx[80], 0)]
poly = Polygon(verts, facecolor='0.9', edgecolor='0.5')
plt.gca().add_patch(poly)

plt.subplot(224)
plt.plot(ss, np.log(ZZ[:, x_idx]), "b-")
max_idx = np.argmax(np.log(ZZ[:, x_idx]))
max_val = np.max(np.log(ZZ[:, x_idx]))
plt.plot(ss[max_idx], max_val, "r.")
plt.plot([ss[max_idx], ss[max_idx]], [-5, max_val], "r:")
plt.plot([0, ss[max_idx]], [max_val, max_val], "r:")
plt.axis([1, 2, -2.4, -2])
plt.xlabel(r"$\theta$", fontsize=14)
plt.text(ss[max_idx]+ 0.01, max_val - 0.05, r"$Max$", fontsize=12)
plt.text(ss[max_idx]+ 0.01, -2.39, r"$\hat{\theta}$", fontsize=14)
plt.text(1.01, max_val + 0.02, r"$\log \, \hat{L}$", fontsize=14)
plt.grid(True)
plt.title(r"$\log \, \mathcal{L}(\theta|x=2.5)$", fontsize=14)

save_fig("likelihood_function_diagram")
plt.show()
