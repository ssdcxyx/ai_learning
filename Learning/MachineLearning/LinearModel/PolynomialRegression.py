# -*- coding: utf-8 -*-
# @Time    : 26/11/18 下午2:18
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : PolynomialRegression.py


import numpy as np
import os
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.linear_model import SGDRegressor
from sklearn.linear_model import Lasso
from sklearn.linear_model import ElasticNet
from sklearn.base import clone
from sklearn import datasets
from sklearn.linear_model import LogisticRegression

np.random.seed(42)

import matplotlib
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)

import warnings
warnings.filterwarnings(action="ignore", module="scipy", message="^internal gelsd")

m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 0.5 * X ** 2 + X + 2 + np.random.randn(m, 1)

# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([-3, 3, 0, 10])
# save_fig("quadratic_data_plot")
# plt.show()

poly_features = PolynomialFeatures(degree=2, include_bias=False)
X_ploy = poly_features.fit_transform(X)
# print(X[0])
# print(X_ploy[0])

lin_reg = LinearRegression()
lin_reg.fit(X_ploy, y)
# print(lin_reg.intercept_, lin_reg.coef_)

X_new = np.linspace(-3, 3, 100).reshape(100, 1)
X_new_poly = poly_features.transform(X_new)
y_new = lin_reg.predict(X_new_poly)
# plt.plot(X, y, "b.")
# plt.plot(X_new, y_new, "r-", linewidth=2, label="Predictions")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=14)
# plt.axis([-3, 3, 0, 10])
# save_fig("quadratic_predictions_plot")
# plt.show()

for style, width, degree in (("g-", 1, 300), ("b--", 2, 2), ("r-+", 2, 1)):
    polybig_features = PolynomialFeatures(degree=degree, include_bias=False)
    std_scaler = StandardScaler()
    lin_reg = LinearRegression()
    polynomial_regression = Pipeline([
        ("poly_features", polybig_features),
        ("std_scaler", std_scaler),
        ("lin_reg", lin_reg)
    ])
    polynomial_regression.fit(X, y)
    y_newbig = polynomial_regression.predict(X_new)
    # plt.plot(X_new, y_newbig, style, label=str(degree), linewidth=width)

# plt.plot(X, y, "b.", linewidth=3)
# plt.legend(loc="upper left")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([-3, 3, 0, 10])
# save_fig("high_degree_polynomials_plt")
# plt.show()


def plot_learning_curves(model, X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2)
    train_errors, val_errors = [], []
    for m in range(1, len(X_train)):
        model.fit(X_train[:m], y_train[:m])
        y_train_predict = model.predict(X_train[:m])
        y_val_predict = model.predict(X_val)
        train_errors.append(mean_squared_error(y_train_predict, y_train[:m]))
        val_errors.append(mean_squared_error(y_val_predict, y_val))
    plt.plot(np.sqrt(train_errors), "r-+", linewidth=2, label="train")
    plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="val")
    plt.legend(loc="upper right", fontsize=14)
    plt.xlabel("Training set size", fontsize=14)
    plt.ylabel("RMSE", fontsize=14)


# lin_reg = LinearRegression()
# plot_learning_curves(lin_reg, X, y)
# plt.axis([0, 80, 0, 3])
# save_fig("underfitting_learning_curves_plot")
# plt.show()

polynomial_regression = Pipeline([
    ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
    ("sgd_reg", LinearRegression())
])

# plot_learning_curves(polynomial_regression, X, y)
# plt.axis([0, 80, 0, 3])
# save_fig("learning_curves_plot")
# plt.show()

# 正则化
m = 20
X = 3 * np.random.rand(m, 1)
y = 1 + 0.5 * X + np.random.randn(m, 1) / 1.5
X_new = np.linspace(0, 3, 100).reshape(100, 1)


def plot_model(model_class, polynomial, alphas, **model_kargs):
    for alpha, style in zip(alphas, ("b-", "g--", "r:")):
        model = model_class(alpha, **model_kargs) if alpha > 0 else LinearRegression()
        if polynomial:
            model = Pipeline([
                ("poly_features", PolynomialFeatures(degree=10, include_bias=False)),
                ("std_scaler", StandardScaler()),
                ("regul_reg", model)
            ])
        model.fit(X, y)
        y_new_regul = model.predict(X_new)
        lw = 2 if alpha > 0 else 1
        plt.plot(X_new, y_new_regul, style, linewidth=lw, label=r"$\alpha ={}$".format(alpha))
    plt.plot(X, y, "b.", linewidth=3)
    plt.legend(loc="upper left", fontsize=15)
    plt.xlabel("$x_1$", fontsize=18)
    plt.axis([0, 3, 0, 4])


# plt.figure(figsize=(8, 4))
# plt.subplot(121)
# plot_model(Ridge, polynomial=False, alphas=(0, 10, 100), random_state=42)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.subplot(122)
# plot_model(Ridge, polynomial=True, alphas=(0, 10**-5, 1), random_state=42)
# save_fig("ridge_regression_plot")
# plt.show()

# l2正则化
# ridge_reg = Ridge(alpha=1, solver="cholesky")
# ridge_reg.fit(X, y)
# print(ridge_reg.predict([[1.5]]))
#
# sgd_reg = SGDRegressor(penalty="l2")
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.predict([[1.5]]))

# plt.figure(figsize=(8, 4))
# plt.subplot(121)
# plot_model(Lasso, polynomial=False, alphas=(0, 0.1, 1), random_state=42)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.subplot(122)
# plot_model(Lasso, polynomial=True, alphas=(0, 10**-7, 1), tol=1, random_state=42)
# save_fig("lasso_regression_plot")
# plt.show()

# l1、l2正则化对比
# t1a, t1b, t2a, t2b = -1, 3, -1.5, 1.5
# t1s = np.linspace(t1a, t1b, 500)
# t2s = np.linspace(t2a, t2b, 500)
# t1, t2 = np.meshgrid(t1s, t2s)
# T = np.c_[t1.ravel(), t2.ravel()]
# Xr = np.array([[-1, 1], [-0.3, -1], [1, 0.1]])
# yr = 2 * Xr[:, :1] + 0.5 * Xr[:, 1:]
#
# J = (1 / len(Xr) * np.sum((T.dot(Xr.T) - yr.T) ** 2, axis=1)).reshape(t1.shape)
#
# N1 = np.linalg.norm(T, ord=1, axis=1).reshape(t1.shape)
# N2 = np.linalg.norm(T, ord=2, axis=1).reshape(t1.shape)
#
# t_min_idx = np.unravel_index(np.argmin(J), J.shape)
# t1_min, t2_min = t1[t_min_idx], t2[t_min_idx]
#
# t_init = np.array([[0.25], [-1]])
#
#
# def bgd_path(theta, X, y, l1, l2, core=1, eta=0.1, n_iterations=50):
#     path = [theta]
#     for iteration in range(n_iterations):
#         gradients = core * 2 / len(X) * X.T.dot(X.dot(theta) - y) + l1 * np.sign(theta) + 2 * l2 * theta
#         theta = theta - eta * gradients
#         path.append(theta)
#     return np.array(path)
#
#
# plt.figure(figsize=(12, 8))
# for i, N, l1, l2, title in ((0, N1, 0.5, 0, "Lasso"), (1, N2, 0, 0.1, "Ridge")):
#     JR = J + l1 * N1 + l2 * N2 ** 2
#
#     tr_min_idx = np.unravel_index(np.argmin(JR), JR.shape)
#     t1r_min, t2r_min = t1[tr_min_idx], t2[tr_min_idx]
#     levelsJ = (np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(J) - np.min(J)) + np.min(J)
#     levelsJR = (np.exp(np.linspace(0, 1, 20)) - 1) * (np.max(JR) - np.min(JR)) + np.min(JR)
#     levelsN = np.linspace(0, np.max(N), 10)
#
#     path_J = bgd_path(t_init, Xr, yr, l1=0, l2=0)
#     path_JR = bgd_path(t_init, Xr, yr, l1, l2)
#     path_N = bgd_path(t_init, Xr, yr, np.sign(l1)/3, np.sign(l2), core=0)
#
#     plt.subplot(221 + i * 2)
#     plt.grid(True)
#     plt.axhline(y=0, color="k")
#     plt.axvline(x=0, color="k")
#     plt.contourf(t1, t2, J, levels=levelsJ, alpha=0.9)
#     plt.contour(t1, t2, N, levels=levelsN)
#     plt.plot(path_J[:, 0], path_J[:, 1], "w-o")
#     plt.plot(path_N[:, 0], path_N[:, 1], "y-^")
#     plt.plot(t1_min, t2_min, "rs")
#     plt.title(r"$\ell_{}$ penalty".format(i + 1), fontsize=16)
#     plt.axis([t1a, t1b, t2a, t2b])
#     if i == 1:
#         plt.xlabel(r"$\theta_1$", fontsize=20)
#     plt.ylabel(r"$\theta_2$", fontsize=20, rotation=0)
#
#     plt.subplot(222 + i * 2)
#     plt.grid(True)
#     plt.axhline(y=0, color="k")
#     plt.axvline(x=0, color="k")
#     plt.contourf(t1, t2, JR, levels=levelsJR, alpha=0.9)
#     plt.plot(path_JR[:, 0], path_JR[:, 1], "w-o")
#     plt.plot(t1r_min, t2r_min, "rs")
#     plt.title(title, fontsize=16)
#     plt.axis([t1a, t1b, t2a, t2b])
#     if i == 1:
#         plt.xlabel(r"$\theta_1$", fontsize=20)
# save_fig("lasso_vs_ridge_plot")
# plt.show()

# l1正则化
# lasso_reg = Lasso(alpha=0.1)
# lasso_reg.fit(X, y)
# print(lasso_reg.predict([[1.5]]))

# sgd_reg = SGDRegressor(penalty="l1")
# sgd_reg.fit(X, y)
# print(sgd_reg.predict([[1.5]]))

# 弹性网络
# elastic_net = ElasticNet(alpha=0.1, l1_ratio=0.5)
# elastic_net.fit(X, y)
# print(elastic_net.predict([[1.5]]))

# 早停法
m = 100
X = 6 * np.random.rand(m, 1) - 3
y = 2 + X + 0.5 * X ** 2 + np.random.randn(m, 1)

X_train, X_val, y_train, y_val = train_test_split(X[:50], y[:50].ravel(), test_size=0.5, random_state=10)

poly_scaler = Pipeline([
    ("poly_features", PolynomialFeatures(degree=90, include_bias=False)),
    ("std_scaler", StandardScaler())
])

X_train_poly_scaled = poly_scaler.fit_transform(X_train)
X_val_poly_scaled = poly_scaler.transform(X_val)

# sgd_reg = SGDRegressor(
#     max_iter=1,
#     penalty=None,
#     eta0=0.0005,
#     warm_start=True,
#     learning_rate="constant",
#     random_state=42
# )
# n_epochs = 500
# train_errors, val_errors = [], []
# for epoch in range(n_epochs):
#     sgd_reg.fit(X_train_poly_scaled, y_train)
#     y_train_predict = sgd_reg.predict(X_train_poly_scaled)
#     y_val_predict = sgd_reg.predict(X_val_poly_scaled)
#     train_errors.append(mean_squared_error(y_train, y_train_predict))
#     val_errors.append(mean_squared_error(y_val, y_val_predict))
#
# best_epoch = np.argmin(val_errors)
# best_val_rmse = np.sqrt(val_errors[best_epoch])
#
# plt.annotate('Best model',
#              xy=(best_epoch, best_val_rmse),
#              xytext=(best_epoch, best_val_rmse+1),
#              ha="center",
#              arrowprops=dict(facecolor='black', shrink=0.05),
#              fontsize=16)
#
# best_val_rmse -= 0.03   # 调整曲线，更美观
# plt.plot([0, n_epochs], [best_val_rmse, best_val_rmse], "k:", linewidth=2)
# plt.plot(np.sqrt(val_errors), "b-", linewidth=3, label="Validation set")
# plt.plot(np.sqrt(train_errors), "r--", linewidth=2, label="Training set")
# plt.legend(loc="upper right", fontsize=14)
# plt.xlabel("Epoch", fontsize=14)
# plt.ylabel("RMSE", fontsize=14)
# save_fig("early_stopping_plot")
# plt.show()

# sgd_reg = SGDRegressor(
#     max_iter=1, warm_start=True, penalty=None,
#     learning_rate="constant", eta0=0.0005, random_state=42
# )
# minimum_val_error = float("inf")
# best_epoch = None
# best_model = None
# for epoch in range(1000):
#     sgd_reg.fit(X_train_poly_scaled, y_train)
#     y_val_predict = sgd_reg.predict(X_val_poly_scaled)
#     val_error = mean_squared_error(y_val, y_val_predict)
#     if val_error < minimum_val_error:
#         minimum_val_error = val_error
#         best_epoch = epoch
#         best_model = clone(sgd_reg)
#
# print(best_epoch, best_model)

# logistic回归
# t = np.linspace(-10, 10, 100)
# sig = 1 / (1+np.exp(-t))
# plt.figure(figsize=(9, 3))
# plt.plot([-10, 10], [0, 0], "k-")
# plt.plot([-10, 10], [0.5, 0.5], "k:")
# plt.plot([-10, 10], [1, 1], "k:")
# plt.plot([-0, 0], [-1.1, 1.1], "k-")
# plt.plot(t, sig, "b-", linewidth=2, label=r"$\sigma(t) = \frac{1}{1+e^{-t}}$")
# plt.xlabel("t")
# plt.legend(loc="upper left", fontsize=20)
# plt.axis([-10, 10, -0.1, 1.1])
# save_fig("logistic_function_plot")
# plt.show()

iris = datasets.load_iris()
# print(iris.keys())
# print(iris.DESCR)

# X = iris["img"][:, 3:]   # petal width
# y = (iris["target"] == 2).astype(np.int)
#
# log_reg = LogisticRegression(random_state=42)
# log_reg.fit(X, y)
#
# X_new = np.linspace(0, 3, 1000).reshape(-1, 1)
# y_proba = log_reg.predict_proba(X_new)

# plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
# plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
# plt.show()

# decision_boundary = X_new[y_proba[:, 1] >= 0.5][0]
#
# plt.figure(figsize=(8, 3))
# plt.plot(X[y == 0], y[y == 0], "bs")
# plt.plot(X[y == 1], y[y == 1], "g^")
# plt.plot([decision_boundary, decision_boundary], [-1, 2], "k:", linewidth=2)
# plt.plot(X_new, y_proba[:, 1], "g-", linewidth=2, label="Iris-Virginica")
# plt.plot(X_new, y_proba[:, 0], "b--", linewidth=2, label="Not Iris-Virginica")
# plt.text(decision_boundary+0.02, 0.15, "Decision boundary", fontsize=14, color="k", ha="center")
# plt.arrow(decision_boundary, 0.08, -0.3, 0, head_width=0.05, head_length=0.1, fc="b", ec="b")
# plt.arrow(decision_boundary, 0.92, 0.3, 0,  head_width=0.05, head_length=0.1, fc="g", ec="g")
# plt.xlabel("Petal width (cm)", fontsize=14)
# plt.ylabel("Probability", fontsize=14)
# plt.legend(loc="center left", fontsize=14)
# plt.axis([0, 3, -0.02, 1.02])
# save_fig("logistic_regression_plot")
# plt.show()

# print(log_reg.predict([[1.7], [1.5]]))

# X = iris["img"][:, (2, 3)]
# y = (iris["target"] == 2).astype(np.int)
#
# log_reg = LogisticRegression(C=10**10, random_state=42)
# log_reg.fit(X, y)
#
# x0, x1 = np.meshgrid(
#     np.linspace(2.9, 7, 500).reshape(-1, 1),
#     np.linspace(0.8, 2.7, 200).reshape(-1, 1)
# )
# X_new = np.c_[x0.ravel(), x1.ravel()]
# y_proba = log_reg.predict_proba(X_new)
#
# plt.figure(figsize=(10, 4))
# plt.plot(X[y == 0, 0], X[y == 0, 1], "bs")
# plt.plot(X[y == 1, 0], X[y == 1, 1], "g^")
# zz = y_proba[:, 1].reshape(x0.shape)
# contour = plt.contour(x0, x1, zz, cmap=plt.cm.brg)
#
# left_right = np.array([2.9, 7])
# boundary = -(log_reg.coef_[0][0] * left_right + log_reg.intercept_[0]) / log_reg.coef_[0][1]
#
# plt.clabel(contour, inline=1, fontsize=12)
# plt.plot(left_right, boundary, "k--", linewidth=3)
# plt.text(3.5, 1.5, "Not Iris-Virginica", fontsize=14, color="b", ha="center")
# plt.text(6.5, 2.3, "Iris-Virginica", fontsize=14, color="g", ha="center")
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.axis([2.9, 7, 0.8, 2.7])
# save_fig("logistic_regression_contour_plot")
# plt.show()

# softmax回归
# X = iris["img"][:, (2, 3)]
# y = iris["target"]
#
# softmax_reg = LogisticRegression(multi_class="multinomial", solver="lbfgs", C=10, random_state=42)
# softmax_reg.fit(X, y)

# x0, x1 = np.meshgrid(
#     np.linspace(0, 8, 500).reshape(-1, 1),
#     np.linspace(0, 3.5, 200).reshape(-1, 1)
# )
# X_new = np.c_[x0.ravel(), x1.ravel()]
# y_proba = softmax_reg.predict_proba(X_new)
# y_predict = softmax_reg.predict(X_new)
#
# zz1 = y_proba[:, 1].reshape(x0.shape)
# zz = y_predict.reshape(x0.shape)
#
# plt.figure(figsize=(10, 4))
# plt.plot(X[y == 2, 0], X[y == 2, 0], "g^", label="Iris-Virginica")
# plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
# plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")
#
# custom_cmap = ListedColormap(["#fafab0", "#9898ff", "#a0faa0"])
#
# plt.contourf(x0, x1, zz, cmap=custom_cmap)
# contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
# plt.clabel(contour, inline=1, fontsize=12)
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal length", fontsize=14)
# plt.legend(loc="center left", fontsize=14)
# plt.axis([0, 7, 0, 3.5])
# save_fig("softmax_regression_contour_plot")
# plt.show()

# print(softmax_reg.predict([[5, 2]]))
# print(softmax_reg.predict_proba([[5, 2]]))

# softmax回归，批量梯度下降，早起停止法
X = iris["img"][:, (2, 3)]
y = iris["target"]

X_with_bias = np.c_[np.ones([len(X), 1]), X]
np.random.seed(2042)

test_ratio = 0.2
validation_ratio = 0.2
total_size = len(X_with_bias)

test_size = int(total_size * test_ratio)
validation_size = int(total_size * validation_ratio)
train_size = total_size - test_size - validation_size

rnd_indices = np.random.permutation(total_size)

X_train = X_with_bias[rnd_indices[:train_size]]
y_train = y[rnd_indices[:train_size]]
X_valid = X_with_bias[rnd_indices[train_size:-test_size]]
y_valid = y[rnd_indices[train_size:-test_size]]
X_test = X_with_bias[rnd_indices[-test_size:]]
y_test = y[rnd_indices[-test_size:]]


def to_one_hot(y):
    n_classes = y.max() + 1
    m = len(y)
    Y_one_hot = np.zeros((m, n_classes))
    Y_one_hot[np.arange(m), y] = 1
    return Y_one_hot


# print(y_train[:10])
# print(to_one_hot(y_train[:10]))

Y_train_one_hot = to_one_hot(y_train)
Y_valid_one_hot = to_one_hot(y_valid)
Y_test_one_hot = to_one_hot(y_test)


def softmax(logits):
    exps = np.exp(logits)
    exp_sums = np.sum(exps, axis=1, keepdims=True)
    return exps / exp_sums


n_inputs = X_train.shape[1]
n_outputs = len(np.unique(y_train))

# eta = 0.01
# n_iterations = 5001
# m = len(X_train)
# epsilon = 1e-7
#
# Theta = np.random.randn(n_inputs, n_outputs)

# for iteration in range(n_iterations):
#     logits = X_train.dot(Theta)
#     Y_proba = softmax(logits)
#     loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
#     error = Y_proba - Y_train_one_hot
#     # if iteration % 500 == 0:
#     #     print(iteration, loss)
#     gradients = 1/m * X_train.T.dot(error)
#     Theta = Theta - eta * gradients

# print(Theta)

# logits = X_valid.dot(Theta)
# Y_proba = softmax(logits)
# y_predict = np.argmax(Y_proba, axis=1)
#
# accuracy_score = np.mean(y_predict == y_valid)
# print(accuracy_score)

# eta = 0.1
# n_iterations = 5001
# m = len(X_train)
# epsilon = 1e-7
# alpha = 0.1
#
# Theta = np.random.randn(n_inputs, n_outputs)
#
#
# for iteration in range(n_iterations):
#     logits = X_train.dot(Theta)
#     Y_proba = softmax(logits)
#     xentropy_loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
#     l2_loss = 1 / 2 * np.sum(np.square(Theta[1:]))
#     loss = xentropy_loss + alpha * l2_loss
#     error = Y_proba - Y_train_one_hot
#     if iteration % 500 == 0:
#         print(iteration, loss)
#     gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha*Theta[1:]]
#     Theta = Theta - eta * gradients
#
# logits = X_valid.dot(Theta)
# Y_proba = softmax(logits)
# y_predict = np.argmax(Y_proba, axis=1)
#
# accuracy_score = np.mean(y_predict == y_valid)
# print(accuracy_score)

eta = 0.1
n_iterations = 5001
m = len(X_train)
epsilon = 1e-7
alpha = 0.1
best_loss = np.infty

Theta = np.random.randn(n_inputs, n_outputs)

for iteration in range(n_iterations):
    logits = X_train.dot(Theta)
    Y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(Y_train_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    error = Y_proba - Y_train_one_hot
    gradients = 1/m * X_train.T.dot(error) + np.r_[np.zeros([1, n_outputs]), alpha * Theta[1:]]
    Theta = Theta - eta * gradients

    logits = X_valid.dot(Theta)
    Y_proba = softmax(logits)
    xentropy_loss = -np.mean(np.sum(Y_valid_one_hot * np.log(Y_proba + epsilon), axis=1))
    l2_loss = 1/2 * np.sum(np.square(Theta[1:]))
    loss = xentropy_loss + alpha * l2_loss
    if iteration % 500 == 0:
        print(iteration, loss)
    if loss < best_loss:
        best_loss = loss
    else:
        print(iteration - 1, best_loss)
        print(iteration, loss, "early stopping")
        break


x0, x1 = np.meshgrid(
    np.linspace(0, 8, 500).reshape(-1, 1),
    np.linspace(0, 3.5, 200).reshape(-1, 1)
)
X_new = np.c_[x0.ravel(), x1.ravel()]
X_new_with_bias = np.c_[np.ones([len(X_new), 1]), X_new]

logits = X_new_with_bias.dot(Theta)
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

zz1 = Y_proba[:, 1].reshape(x0.shape)
zz = y_predict.reshape(x0.shape)

plt.figure(figsize=(10, 4))
plt.plot(X[y == 2, 0], X[y == 2, 1], "g^", label="Iris-Virginica")
plt.plot(X[y == 1, 0], X[y == 1, 1], "bs", label="Iris-Versicolor")
plt.plot(X[y == 0, 0], X[y == 0, 1], "yo", label="Iris-Setosa")

custom_cmap = ListedColormap(['#fafab0', '#9898ff', '#a0faa0'])

plt.contourf(x0, x1, zz, cmap=custom_cmap)
contour = plt.contour(x0, x1, zz1, cmap=plt.cm.brg)
plt.clabel(contour, inline=1, fontsize=12)
plt.xlabel("Petal length", fontsize=14)
plt.ylabel("Petal width", fontsize=14)
plt.legend(loc="upper left", fontsize=14)
plt.axis([0, 7, 0, 3.5])
plt.show()

ƒn
Y_proba = softmax(logits)
y_predict = np.argmax(Y_proba, axis=1)

accuracy_score = np.mean(y_predict == y_test)
print(accuracy_score)
