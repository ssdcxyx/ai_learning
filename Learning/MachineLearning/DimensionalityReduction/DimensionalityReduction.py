# -*- coding: utf-8 -*-
# @Time    : 2018-12-13 20:56
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : DimensionalityReduction.py

import numpy as np
import os
from sklearn.decomposition import PCA
from matplotlib.patches import FancyArrowPatch
from mpl_toolkits.mplot3d import proj3d
from mpl_toolkits.mplot3d import Axes3D
from sklearn.datasets import make_swiss_roll
from matplotlib import gridspec
from six.moves import urllib
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split
from sklearn.decomposition import IncrementalPCA
import time
from sklearn.decomposition import KernelPCA
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.manifold import LocallyLinearEmbedding
from sklearn.manifold import MDS
from sklearn.manifold import Isomap
from sklearn.manifold import TSNE
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import MinMaxScaler
from matplotlib.offsetbox import AnnotationBbox, OffsetImage

np.random.seed(42)

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "unsupervised_learning"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, 'images', CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

# PCA 主成分分析
np.random.seed(4)
m = 60
w1, w2 = 0.1, 0.3
noise = 0.1

angles = np.random.rand(m) * 3 * np.pi / 2 - 0.5
X = np.empty((m, 3))
X[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * np.random.randn(m) / 2
X[:, 1] = np.sin(angles) * 0.7 + noise * np.random.randn(m) / 2
X[:, 2] = X[:, 0] * w1 + X[:, 1] * w2 + noise * np.random.randn(m)

X_centered = X - X.mean(axis=0)
U, s, Vt = np.linalg.svd(X_centered)
c1 = Vt.T[:, 0]
c2 = Vt.T[:, 1]

m, n = X.shape
S = np.zeros(X_centered.shape)
S[:n, :n] = np.diag(s)
#
# print(np.allclose(X_centered, U.dot(S).dot(Vt)))
#
W2 = Vt.T[:, :2]
X2D = X_centered.dot(W2)
X2D_using_svd = X2D

pca = PCA(n_components=2)
X2D = pca.fit_transform(X)

# print(X2D[:5])
# print(X2D_using_svd[:5])

# print(np.allclose(X2D, -X2D_using_svd))
#
X3D_inv = pca.inverse_transform(X2D)
# print(np.allclose(X3D_inv, X))
#
# print(np.mean(np.sum(np.square(X3D_inv - X), axis=1)))
#
X3D_inv_using_svd = X2D_using_svd.dot(Vt[:2, :])
#
# print(np.allclose(X3D_inv_using_svd, X3D_inv - pca.mean_))
#
# print(pca.components_)
#
# print(Vt[:2])
#
# print(pca.explained_variance_ratio_)
#
# print(1 - pca.explained_variance_ratio_.sum())
#
# print(np.square(s) / np.square(s).sum())


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        FancyArrowPatch.__init__(self, (0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj3d.proj_transform(xs3d, ys3d, zs3d, renderer.M)
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        FancyArrowPatch.draw(self, renderer)


axes = [-1.8, 1.8, -1.3, 1.3, -1.0, 1.0]
x1s = np.linspace(axes[0], axes[1], 10)
x2s = np.linspace(axes[2], axes[3], 10)
x1, x2 = np.meshgrid(x1s, x2s)

C = pca.components_
R = C.T.dot(C)
z = (R[0, 2] * x1 + R[1, 2] * x2) / (1 - R[2, 2])

# fig = plt.figure(figsize=(6, 3.8))
# ax = fig.add_subplot(111, projection='3d')
# X3D_above = X[X[:, 2] > X3D_inv[:, 2]]
# X3D_below = X[X[:, 2] <= X3D_inv[:, 2]]
#
# ax.plot(X3D_below[:, 0], X3D_below[:, 1], X3D_below[:, 2], "bo", alpha=0.5)
#
# ax.plot_surface(x1, x2, z, alpha=0.2, color="k")
# np.linalg.norm(C, axis=0)
# ax.add_artist(Arrow3D([0, C[0, 0]], [0, C[0, 1]], [0, C[0, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
# ax.add_artist(Arrow3D([0, C[1, 0]], [0, C[1, 1]], [0, C[1, 2]], mutation_scale=15, lw=1, arrowstyle="-|>", color="k"))
# ax.plot([0], [0], [0], "k.")
#
# for i in range(m):
#     if X[i, 2] > X3D_inv[i, 2]:
#         ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k--")
#     else:
#         ax.plot([X[i][0], X3D_inv[i][0]], [X[i][1], X3D_inv[i][1]], [X[i][2], X3D_inv[i][2]], "k--", color="#505050")
#
# ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k+")
# ax.plot(X3D_inv[:, 0], X3D_inv[:, 1], X3D_inv[:, 2], "k.")
# ax.plot(X3D_above[:, 0], X3D_above[:, 1], X3D_above[:, 2], "bo")
# ax.set_xlabel("$x_1$", fontsize=18)
# ax.set_ylabel("$x_2$", fontsize=18)
# ax.set_zlabel("$x_3$", fontsize=18)
# ax.set_xlim(axes[0: 2])
# ax.set_ylim(axes[2: 4])
# ax.set_zlim(axes[4: 6])

# save_fig("dataset_3d_plot")
# plt.show()

# fig = plt.figure()
# ax = fig.add_subplot(111, aspect="equal")
#
# ax.plot(X2D[:, 0], X2D[:, 1], "k+")
# ax.plot(X2D[:, 0], X2D[:, 1], "k.")
# ax.plot([0], [0], "ko")
# ax.arrow(0, 0, 0, 1, head_width=0.05, length_includes_head=True, head_length=0.1, fc="k", ec="k")
# ax.arrow(0, 0, 1, 0, head_width=0.05, length_includes_head=True, head_length=0.1, fc="k", ec="k")
#
# ax.set_xlabel("$z_1$", fontsize=18)
# ax.set_ylabel("$z_2$", fontsize=18, rotation=0)
# ax.axis([-1.5, 1.3, -1.2, 1.2])
# ax.grid(True)
# save_fig("dataset_2d_plot")

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

axes = [-11.5, 14, -2, 23, -12, 15]

# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111, projection='3d')

# ax.scatter(X[:, 0], X[:, 1], X[:, 2], c=t, cmap=plt.cm.hot)
# ax.view_init(10, -70)
# ax.set_xlabel("$x_1$")
# ax.set_ylabel("$y_1$")
# ax.set_zlabel("$z_1$")
# ax.set_xlim(axes[0:2])
# ax.set_ylim(axes[2:4])
# ax.set_zlim(axes[4:6])
#
# save_fig("swiss_roll_plot")
# plt.show()

# plt.figure(figsize=(11, 4))
#
# plt.subplot(121)
# plt.scatter(X[:, 0], X[:, 1], c=t, cmap=plt.cm.hot)
# plt.axis(axes[:4])
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$x_2$", fontsize=18, rotation=0)
# plt.grid(True)
#
# plt.subplot(122)
# plt.scatter(t, X[:, 1], c=t, cmap=plt.cm.hot)
# plt.axis([4, 15, axes[2], axes[3]])
# plt.xlabel("$z_1$", fontsize=18)
# plt.grid(True)
#
# save_fig("squished_swiss_roll_plot")
# plt.show()

axes = [-11.5, 14, -2, 23, -12, 15]

x2s = np.linspace(axes[2], axes[3], 10)
x3s = np.linspace(axes[4], axes[5], 10)
x2, x3 = np.meshgrid(x2s, x3s)

# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111, projection='3d')
#
positive_class = X[:, 0] > 5
X_pos = X[positive_class]
X_neg = X[~positive_class]
# ax.view_init(10, -70)
# ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
# ax.plot_wireframe(5, x2, x3, alpha=0.5)
# ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
# ax.set_xlabel("$x_1$")
# ax.set_ylabel("$x_2$")
# ax.set_zlabel("$x_3$")
# ax.set_xlim(axes[0:2])
# ax.set_ylim(axes[2:4])
# ax.set_zlim(axes[4:6])
#
# save_fig("manifold_decision_boundary_plot1")
# plt.show()


# fig = plt.figure(figsize=(5, 4))
#
# plt.plot(t[positive_class], X[positive_class], "gs")
# plt.plot(t[~positive_class], X[~positive_class], "y^")
# plt.axis([4, 15, axes[2], axes[3]])
# plt.xlabel("$z_1$", fontsize=18)
# plt.ylabel("$z_2$", fontsize=18, rotation=0)
# plt.grid(True)
#
# save_fig("manifold_decision_boundary_plot2")
# plt.show()

# fig = plt.figure(figsize=(6, 5))
# ax = fig.add_subplot(111, projection='3d')
#
positive_class = 2 * (t[:] - 4) > X[:, 1]
X_pos = X[positive_class]
X_neg = X[~positive_class]
# ax.view_init(10, -70)
# ax.plot(X_neg[:, 0], X_neg[:, 1], X_neg[:, 2], "y^")
# ax.plot(X_pos[:, 0], X_pos[:, 1], X_pos[:, 2], "gs")
# ax.set_xlabel("$x_1$")
# ax.set_ylabel("$x_2$")
# ax.set_zlabel("$x_3$")
# ax.set_xlim(axes[0:2])
# ax.set_ylim(axes[2:4])
# ax.set_zlim(axes[4:6])
#
# save_fig("manifold_decision_boundary_plot3")
# plt.show()

# fig = plt.figure(figsize=(5, 4))
# ax = plt.subplot(111)
#
# plt.plot(t[positive_class], X[positive_class, 1], "gs")
# plt.plot(t[~positive_class], X[~positive_class, 1], "y^")
# plt.plot([4, 15], [0, 22], "b-", linewidth=2)
# plt.axis([4, 15, axes[2], axes[3]])
# plt.xlabel("$z_1$", fontsize=18)
# plt.ylabel("$z_2$", fontsize=18, rotation=0)
# plt.grid(True)
#
# save_fig("manifold_decision_boundary_plot4")
# plt.show()

# angle = np.pi / 5
# stretch = 5
# m = 200
#
# np.random.seed(3)
# X = np.random.randn(m, 2) / 10
# X = X.dot(np.array([[stretch, 0], [0, 1]]))
# X = X.dot([[np.cos(angle), np.sin(angle)], [-np.sin(angle), np.cos(angle)]])
#
# u1 = np.array([np.cos(angle), np.sin(angle)])
# u2 = np.array([np.cos(angle - 2 * np.pi * 6), np.sin(angle - 2 * np.pi / 6)])
# u3 = np.array([np.cos(angle - np.pi / 2), np.sin(angle - np.pi / 2)])
#
# X_proj1 = X.dot(u1.reshape(-1, 1))
# X_proj2 = X.dot(u2.reshape(-1, 1))
# X_proj3 = X.dot(u3.reshape(-1, 1))
#
# plt.figure(figsize=(8, 4))
# plt.subplot2grid((3, 2), (0, 0), rowspan=3)
# plt.plot([-1.4, 1.4], [-1.4 * u1[1] / u1[0], 1.4 * u1[1] / u1[0]], "k-", linewidth=1)
# plt.plot([-1.4, 1.4], [-1.4 * u2[1] / u2[0], 1.4 * u2[1] / u2[0]], "k--", linewidth=1)
# plt.plot([-1.4, 1.4], [-1.4 * u3[1] / u3[0], 1.4 * u3[1] / u3[0]], "k:", linewidth=2)
# plt.plot(X[:, 0], X[:, 1], "bo", alpha=0.5)
# plt.axis([-1.4, 1.4, -1.4, 1.4])
# plt.arrow(0, 0, u1[0], u1[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc="k", ec="k")
# plt.arrow(0, 0, u3[0], u3[1], head_width=0.1, linewidth=5, length_includes_head=True, head_length=0.1, fc="k", ec="k")
# plt.text(u1[0] + 0.1, u1[1] - 0.05, r"$\mathbf{c_1}$", fontsize=22)
# plt.text(u3[0] + 0.1, u3[1], r"$\mathbf{c_2}$", fontsize=22)
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$x_2$", fontsize=18, rotation=0)
# plt.grid(True)
#
# plt.subplot2grid((3, 2), (0, 1))
# plt.plot([-2, 2], [0, 0], "k-", linewidth=1)
# plt.plot(X_proj1[:, 0], np.zeros(m), "bo", alpha=0.3)
# plt.gca().get_yaxis().set_ticks([])
# plt.gca().get_yaxis().set_ticklabels([])
# plt.axis([-2, 2, -1, 1])
# plt.grid(True)
#
# plt.subplot2grid((3, 2), (1, 1))
# plt.plot([-2, 2], [0, 0], "k--", linewidth=1)
# plt.plot(X_proj2[:, 0], np.zeros(m), "bo", alpha=0.3)
# plt.gca().get_yaxis().set_ticks([])
# plt.gca().get_yaxis().set_ticklabels([])
# plt.axis([-2, 2, -1, 1])
# plt.grid(True)
#
# plt.subplot2grid((3, 2), (2, 1))
# plt.plot([-2, 2], [0, 0], "k:", linewidth=2)
# plt.plot(X_proj3[:, 0], np.zeros(m), "bo", alpha=0.3)
# plt.gca().get_yaxis().set_ticks([])
# plt.gca().get_yaxis().set_ticklabels([])
# plt.axis([-2, 2, -1, 1])
# plt.grid(True)
#
# save_fig("pca_best_projection")
# plt.show()

mnist = fetch_mldata("MNIST original")

X = mnist["img"]
y = mnist["target"]

X_train, X_test, y_train, y_test = train_test_split(X, y)

# pca = PCA()
# pca.fit(X_train)
# cumsum = np.cumsum(pca.explained_variance_ratio_)
# d = np.argmax(cumsum >= 0.95) + 1
#
# print(d)
#
# pca = PCA(n_components=0.95)
# X_reduced = pca.fit_transform(X_train)
# print(pca.n_components_)
#
# print(np.sum(pca.explained_variance_ratio_))

# pca = PCA(n_components=154)
# X_reduced = pca.fit_transform(X_train)
# X_recovered = pca.inverse_transform(X_reduced)


def plot_digits(instances, images_per_row=5, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    n_empty = n_rows * images_per_row - len(instances)
    images.append(np.zeros((size, size * n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row:(row+1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")


# plt.figure(figsize=(7, 4))
# plt.subplot(121)
# plot_digits(X_train[::2100])
# plt.title("Original", fontsize=16)
# plt.subplot(122)
# plot_digits(X_recovered[::2100])
# plt.title("Compressed", fontsize=16)
#
# save_fig("mnist_compression_plot")
# plt.show()

# X_reduced_pca = X_reduced

# 增量PCA
# n_batches = 100
# inc_pca = IncrementalPCA(n_components=154)
# for X_batch in np.array_split(X_train, n_batches):
#     # print(".", end="")
#     inc_pca.partial_fit(X_batch)
#
# X_reduced = inc_pca.transform(X_train)
# X_recovered_inc_pca = inc_pca.inverse_transform(X_reduced)

# plt.figure(figsize=(7, 4))
# plt.subplot(121)
# plot_digits(X_train[::2100])
# plt.title("Original", fontsize=16)
# plt.subplot(122)
# plot_digits(X_recovered[::2100])
# plt.title("Compressed", fontsize=16)
# plt.show()

# X_reduced_inc_pca = X_reduced

# print(np.allclose(pca.mean_, inc_pca.mean_))
# print(np.allclose(X_reduced_pca, X_reduced_inc_pca))

# filename = "my_mnist.img"
# m, n = X_train.shape
# X_mm = np.memmap(filename, dtype="float32", mode="write", shape=(m, n))
# X_mm[:] = X_train
#
# del X_mm
#
# X_mm = np.memmap(filename, dtype="float32", mode="readonly", shape=(m, n))
# batch_size = m // n_batches
# inc_pca = IncrementalPCA(n_components=154, batch_size=batch_size)
# print(inc_pca.fit(X_mm))

# 随机PCA
# rnd_pca = PCA(n_components=154, svd_solver="randomized", random_state=42)
# X_reduced = rnd_pca.fit_transform(X_train)


# for n_components in (2, 10, 154):
#     print("n_components = ", n_components)
#     regular_pca = PCA(n_components=n_components)
#     inc_pca = IncrementalPCA(n_components=n_components, batch_size=500)
#     rnd_pca = PCA(n_components=n_components, random_state=42, svd_solver="randomized")
#
#     for pca in (regular_pca, inc_pca, rnd_pca):
#         t1 = time.time()
#         pca.fit(X_train)
#         t2 = time.time()
#         print(" {}:{:.1f} seconds".format(pca.__class__.__name__, t2 - t1))

# times_rpca = []
# times_pca = []
# sizes = [1000, 10000, 20000, 30000, 40000, 70000, 10000, 200000, 500000]
# for n_samples in sizes:
#     X = np.random.randn(n_samples, 5)
#     pca = PCA(n_components=2, svd_solver="randomized", random_state=42)
#     t1 = time.time()
#     pca.fit(X)
#     t2 = time.time()
#     times_rpca.append(t2 - t1)
#     pca = PCA(n_components=2)
#     t1 = time.time()
#     pca.fit(X)
#     t2 = time.time()
#     times_pca.append(t2 - t1)
#
# plt.plot(sizes, times_rpca, "b-o", label="RPCA")
# plt.plot(sizes, times_pca, "r-s", label="PCA")
# plt.xlabel("n_samples")
# plt.ylabel("Traning time")
# plt.legend(loc="upper left")
# plt.title("PCA and Randomized PCA time complexity")
# plt.show()

# times_rpca = []
# times_pca = []
# sizes = [1000, 2000, 3000, 4000, 5000, 6000]
#
# for n_features in sizes:
#     X = np.random.randn(2000, n_features)
#     pca = PCA(n_components=2, random_state=42, svd_solver="randomized")
#     t1 = time.time()
#     pca.fit(X)
#     t2 = time.time()
#     times_rpca.append(t2 - t1)
#     pca = PCA(n_components=2)
#     t1 = time.time()
#     pca.fit(X)
#     t2 = time.time()
#     times_pca.append(t2 - t1)
#
# plt.plot(sizes, times_rpca, "b-o", label="PCA")
# plt.plot(sizes, times_pca, "r-s", label="PCA")
# plt.xlabel("n_features")
# plt.ylabel("Training time")
# plt.legend(loc="upper left")
# plt.title("PCA and Randomized PCA time complexity")
# plt.show()

X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=42)

rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.04)
X_reduced = rbf_pca.fit_transform(X)

lin_pca = KernelPCA(n_components=2, kernel="linear", fit_inverse_transform=True)
rbf_pca = KernelPCA(n_components=2, kernel="rbf", gamma=0.0433, fit_inverse_transform=True)
sig_pca = KernelPCA(n_components=2, kernel="sigmoid", gamma=0.001, coef0=1, fit_inverse_transform=True)

y = t > 6.9

# plt.figure(figsize=(11, 4))
for subplot, pca, title in ((131, lin_pca, "Linear kernel"),
                            (132, rbf_pca, "RBF kernel, $\gamma=0.04$"),
                            (133, sig_pca, "Sigmoid kernel, $\gamma=10^{-3}, r=1$")):
    X_reduced = pca.fit_transform(X)
    if subplot == 132:
        X_reduced_rbf = X_reduced
    # plt.subplot(subplot)
    # plt.title(title, fontsize=14)
    # plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
    # plt.xlabel("$z_1$", fontsize=18)
    # if subplot == 131:
    #     plt.ylabel("$z_2$", fontsize=18, rotation=0)
    # plt.grid(True)

# save_fig("kernel_pca_plot")
# plt.show()

# fig = plt.figure(figsize=(6, 5))
# # X_inverse = rbf_pca.inverse_transform(X_reduced_rbf)
# #
# # ax = fig.add_subplot(111, projection='3d')
# # ax.view_init(10, -70)
# # ax.scatter(X_inverse[:, 0], X_inverse[:, 1], X_inverse[:, 2], c=t, cmap=plt.cm.hot, marker="x")
# # ax.set_xlabel("")
# # ax.set_ylabel("")
# # ax.set_zlabel("")
# # ax.set_xticklabels([])
# # ax.set_yticklabels([])
# # ax.set_zticklabels([])
# #
# # save_fig("preimage_plot", tight_layout=False)
# # plt.show()

# X_reduced = rbf_pca.fit_transform(X)
#
# plt.figure(figsize=(11, 4))
# plt.subplot(132)
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot, marker="x")
# plt.xlabel("$z_1$", fontsize=18)
# plt.ylabel("$z_2$", fontsize=18, rotation=0)
# plt.grid(True)
# plt.show()

# clf = Pipeline([
#     ("kpca", KernelPCA(n_components=2)),
#     ("log_reg", LogisticRegression())
# ])
#
# param_grid = [{
#     "kpca__kernel": ["rbf", "sigmoid"],
#     "kpca__gamma": np.linspace(0.03, 0.05, 10)
# }]
#
# grid_search = GridSearchCV(clf, param_grid, cv=3)
# grid_search.fit(X, y)
#
# print(grid_search.best_estimator_)
#
# rbf_pca = KernelPCA(n_components=2, kernel='rbf', gamma=0.0433, fit_inverse_transform=True)
# X_reduced = rbf_pca.fit_transform(X)
# X_preimage = rbf_pca.inverse_transform(X_reduced)
#
# print(mean_squared_error(X, X_preimage))

# X, t = make_swiss_roll(n_samples=1000, noise=0.2, random_state=41)
#
# lle = LocallyLinearEmbedding(n_components=2, n_neighbors=10, random_state=42)
# X_reduced = lle.fit_transform(X)
#
# plt.title("Unroll swiss roll using LLE", fontsize=14)
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
# plt.xlabel("$z_1$", fontsize=18)
# plt.ylabel("$z_2$", fontsize=18)
# plt.axis([-0.065, 0.055, -0.1, 0.12])
# plt.grid(True)
# save_fig("lle_unrolling_plot")
# plt.show()

# MDS 多维缩放
# mds = MDS(n_components=2, random_state=42)
# X_reduced_mds = mds.fit_transform(X)

# isomap
# isomap = Isomap(n_components=2)
# X_reduced_isomap = isomap.fit_transform(X)

# t-SNE t分布式随机领域嵌入
# tsne = TSNE(n_components=2, random_state=42)
# X_reduced_tsne = tsne.fit_transform(X)

# 线性判别分析
# lda = LinearDiscriminantAnalysis(n_components=2)
# X_mnist = mnist["img"]
# y_mnist = mnist["target"]
# lda.fit(X_mnist, y_mnist)
# X_reduced_lda = lda.transform(X_mnist)

# titles = ["MDS", "Isomap", "t-SNE"]
# plt.figure(figsize=(11, 4))
#
# for subplot, title, X_reduced in zip((131, 132, 133), titles, (X_reduced_mds, X_reduced_isomap, X_reduced_tsne)):
#     plt.subplot(subplot)
#     plt.title(title, fontsize=14)
#     plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=t, cmap=plt.cm.hot)
#     plt.xlabel("$z_1$", fontsize=18)
#     if subplot == 131:
#         plt.ylabel("$z_2$", fontsize=18, rotation=0)
#     plt.grid(True)
#
# save_fig("other_dim_reduction_plot")
# plt.show()

# mnist = fetch_mldata("MNIST original")
#
# X_train = mnist['img'][:60000]
# y_train = mnist['target'][:60000]
#
# X_test = mnist['img'][60000:]
# y_test = mnist['target'][60000:]
#
# rnd_clf = RandomForestClassifier(random_state=42)

# t0 = time.time()
# rnd_clf.fit(X_train, y_train)
# t1 = time.time()
# print("Training took {:.2f}s".format(t1 - t0))
# y_pred = rnd_clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))

# pca = PCA(n_components=0.95)
# X_train_reduced = pca.fit_transform(X_train)

# rnd_clf2 = RandomForestClassifier(random_state=42)
# t0 = time.time()
# rnd_clf2.fit(X_train_reduced, y_train)
# t1 = time.time()
# print("Traing look {:.2f}s".format(t1 - t0))
#
# X_test_reduced = pca.transform(X_test)
# y_pred = rnd_clf2.predict(X_test_reduced)
# print(accuracy_score(y_test, y_pred))

# log_clf = LogisticRegression(multi_class="multinomial", solver='lbfgs', random_state=42)
# t0 = time.time()
# log_clf.fit(X_train, y_train)
# t1 = time.time()
# print("Training took {:.2f}s".format(t1 - t0))
# y_pred = log_clf.predict(X_test)
# print(accuracy_score(y_test, y_pred))
#
# log_clf2 = LogisticRegression(multi_class="multinomial", solver="lbfgs", random_state=42)
# t0 = time.time()
# log_clf2.fit(X_train_reduced, y_train)
# t1 = time.time()
# print("Training took {:.2f}s".format(t1 - t0))
# y_pred = log_clf2.predict(X_test_reduced)
# print(accuracy_score(y_test, y_pred))

mnist = fetch_mldata("MNIST original")

np.random.seed(42)

m = 10000
idx = np.random.permutation(60000)[:m]

X = mnist['img'][idx]
y = mnist['target'][idx]

# t-SNE t分布式随机领域嵌入
# tsne = TSNE(n_components=2, random_state=42)
# X_reduced = tsne.fit_transform(X)

# plt.figure(figsize=(13, 10))
# plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, cmap="jet")
# plt.axis("off")
# plt.colorbar()
# plt.show()

# plt.figure(figsize=(9, 9))
# cmap = matplotlib.cm.get_cmap("jet")
# for digit in (2, 3, 5):
#     plt.scatter(X_reduced[y == digit, 0], X_reduced[y == digit, 1], c=cmap(digit/9))
# plt.axis("off")
# plt.show()

# idx = (y == 2) | (y == 3) | (y == 5)
# X_subset = X[idx]
# y_subset = y[idx]
#
# tsne_subset = TSNE(n_components=2, random_state=42)
# X_subset_reduced = tsne_subset.fit_transform(X_subset, y_subset)
# plt.figure(figsize=(9, 9))
# cmap = matplotlib.cm.get_cmap("jet")
# for digit in (2, 3, 5):
#     plt.scatter(X_subset_reduced[y_subset == digit, 0], X_subset_reduced[y_subset == digit, 1], c=cmap(digit/9))
# plt.axis("off")
# plt.show()


def plot_digits(X, y, min_distance=0.05, images=None, figsize=(13, 10)):
    X_normalized = MinMaxScaler().fit_transform(X)
    neighbors = np.array([[10., 10]])
    plt.figure(figsize=figsize)
    cmap = matplotlib.cm.get_cmap("jet")
    digits = np.unique(y)
    for digit in digits:
        plt.scatter(X_normalized[y == digit, 0], X_normalized[y == digit, 1], c=cmap(digit/9))
    plt.axis("off")
    ax = plt.gcf().gca()
    for index, image_coord in enumerate(X_normalized):
        closest_distance = np.linalg.norm(np.array(neighbors) - image_coord, axis=1).min()
        if closest_distance > min_distance:
            neighbors = np.r_[neighbors, [image_coord]]
            if images is None:
                plt.text(image_coord[0], image_coord[1], str(int(y[index])),
                         color=cmap(y[index] / 9), fontdict={"weight": "bold", "size": 16})
            else:
                image = images[index].reshape(28, 28)
                imagebox = AnnotationBbox(OffsetImage(image, cmap="binary"), image_coord)
                ax.add_artist(imagebox)


# plot_digits(X_reduced, y)

# plot_digits(X_reduced, y, images=X, figsize=(35, 25))

# plot_digits(X_subset_reduced, y_subset, images=X_subset, figsize=(22, 22))

# PCA 主成分分析
# t0 = time.time()
# X_pca_reduced = PCA(n_components=2, random_state=42).fit_transform(X)
# t1 = time.time()
# print("PCA took {:.1f}s".format(t1 - t0))
# plot_digits(X_pca_reduced, y)

# LLE 局部线性嵌入
# t0 = time.time()
# X_lle_reduced = LocallyLinearEmbedding(n_components=2, random_state=42).fit_transform(X)
# t1 = time.time()
# print("LLE took {:.1f}s".format(t1 - t0))
# plot_digits(X_lle_reduced, y)

# PCA + LLE
# pca_lle = Pipeline([
#     ("pca", PCA(n_components=0.95, random_state=42)),
#     ("lle", LocallyLinearEmbedding(n_components=2, random_state=42))
# ])
# t0 = time.time()
# X_pca_lle_reduced = pca_lle.fit_transform(X)
# t1 = time.time()
# print("PCA+LLE took {:.1f}s".format(t1 - t0))
# plot_digits(X_pca_lle_reduced, y)

# MDS 多维缩放
# m = 2000
# t0 = time.time()
# X_mds_reduced = MDS(n_components=2, random_state=42).fit_transform(X[:m])
# t1 = time.time()
# print("MDS took {:.1f}s on 2000 MNIST images".format(t1 - t0))
# plot_digits(X_mds_reduced, y[:m])

# PCA + MDS
# pca_mds = Pipeline([
#     ("pca", PCA(n_components=0.95, random_state=42)),
#     ("mds", MDS(n_components=2, random_state=42))
# ])
# t0 = time.time()
# X_pca_mds_reduced = pca_mds.fit_transform(X[:2000])
# t1 = time.time()
# print("PCA+MDS took {:.1f}s on 2000 MNIST images".format(t1 - t0))
# plot_digits(X_pca_mds_reduced, y[:2000])

# LDA 线性判别分析
# t0 = time.time()
# X_lda_reduced = LinearDiscriminantAnalysis(n_components=2).fit_transform(X, y)
# t1 = time.time()
# print("LDA took {:.1f}s".format(t1 - t0))
# plot_digits(X_lda_reduced, y, figsize=(12, 12))

# t-NSE t分布式随机领域嵌入
# t0 = time.time()
# X_tsne_reduced = TSNE(n_components=2, random_state=42).fit_transform(X)
# t1 = time.time()
# print("t-SNE took {:.1f}s".format(t1 - t0))
# plot_digits(X_tsne_reduced, y)

# PCA + t-NSE
pca_tsne = Pipeline([
    ("pca", PCA(n_components=0.95, random_state=42)),
    ("tsne", TSNE(n_components=2, random_state=42))
])
t0 = time.time()
X_pca_tsne_reduced = pca_tsne.fit_transform(X)
t1 = time.time()
print("PCA + t-NSE took {:.1f}s".format(t1 - t0))
plot_digits(X_pca_tsne_reduced, y)

plt.show()

