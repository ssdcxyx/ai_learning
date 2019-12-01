# -*- coding: utf-8 -*-
# @Time    : 2018/11/14 上午8:39
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : LinearRegression.py


import numpy as np
import os
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor

np.random.seed(42)

import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "training_linear_models"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id+".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


import warnings
warnings.filterwarnings(action='ignore', module="scipy", message="^internal gelsd")

X = 2 * np.random.rand(100, 1)
y = 4 + 3 * X + np.random.randn(100, 1)

# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([0, 2, 0, 15])
# save_fig("generated_data_plot")
# plt.show()

# 正态方程求解
X_b = np.c_[np.ones((4, 1)), X]
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
# print(theta_best)

X_new = np.array([[0], [2]])
X_new_b = np.c_[np.ones((2, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
# print(y_predict)

# plt.plot(X_new, y_predict, "r-")
# plt.plot(X, y, "b.")
# plt.axis([0, 2, 0, 15])
# plt.show()

lin_reg = LinearRegression()
lin_reg.fit(X, y)
# print(lin_reg.intercept_, lin_reg.coef_)
# print(lin_reg.predict(X_new))

# eta = 0.1  # 学习率
# n_iterations = 100
# m = 100
#
# theta = np.random.randn(2, 1)

# for iteration in range(n_iterations):
#     gradients = 2 / m * X_b.T.dot(X_b.dot(theta) - y)
#     theta = theta - eta * gradients

# print(theta)

theta = np.random.randn(2, 1)

theta_path_bgd = []


# 批量梯度下降
def plot_gradient_descent(theta, eta, theta_path=None):
    m = len(X_b)
    # plt.plot(X, y, "b.")
    n_iterations = 1000
    for iteration in range(n_iterations):
        if iteration < 10:
            y_predict = X_new_b.dot(theta)
            style = "b-" if iteration > 0 else 'r--'
            # plt.plot(X_new, y_predict, style)
        gradients = 2/m * X_b.T.dot(X_b.dot(theta) - y)
        theta = theta - eta*gradients
        if theta_path is not None:
            theta_path.append(theta)
    # plt.xlabel("$x_1$", fontsize=18)
    # plt.axis([0, 2, 0, 15])
    # plt.title(r'$\eta = {}$'.format(eta), fontsize=16)


# plt.figure(figsize=(10, 4))
# plt.subplot(131); plot_gradient_descent(theta, eta=0.02)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.subplot(132);
plot_gradient_descent(theta, eta=0.1, theta_path=theta_path_bgd)
# plt.subplot(133); plot_gradient_descent(theta, eta=0.5)
# save_fig("gradient_descent_plot")
# plt.show()

theta_path_sgd = []
m = len(X_b)
n_epochs = 50
t0, t1 = 5, 50


def learning_schedule(t):
    return t0 / (t + t1)


# 随机梯度下降
for epoch in range(n_epochs):
    for i in range(m):
        if epoch == 0 and i < 20:
            y_predict = X_new_b.dot(theta)
            style = "b--" if i > 0 else "r--"
            # plt.plot(X_new, y_predict, style)
        random_index = np.random.randint(m)
        xi = X_b[random_index:random_index+1]
        yi = y[random_index:random_index+1]
        gradients = 2*xi.T.dot(xi.dot(theta)-yi)
        eta = learning_schedule(epoch*m+i)
        theta = theta - eta*gradients
        theta_path_sgd.append(theta)

# plt.plot(X, y, "b.")
# plt.xlabel("$x_1$", fontsize=18)
# plt.ylabel("$y$", rotation=0, fontsize=18)
# plt.axis([0, 2, 0, 15])
# save_fig("sgd_plot")
# plt.show()

# print(theta)

# sgd_reg = SGDRegressor(max_iter=50, penalty=None, eta0=0.1, random_state=42)
# sgd_reg.fit(X, y.ravel())
# print(sgd_reg.intercept_)
# print(sgd_reg.coef_)

theta_path_mgd = []

n_iterations = 50
minibatch_size = 20

theta = np.random.randn(2, 1)
t0, t1 = 200, 1000


def learning_schedule(t):
    return t0 / (t + t1)


# 小批量随机下降
t = 0
for epoch in range(n_iterations):
    shuffled_indices = np.random.permutation(m)
    X_b_shuffled = X_b[shuffled_indices]
    X_b_shuffled = X_b[shuffled_indices]
    y_shuffled = y[shuffled_indices]
    for i in range(0, m, minibatch_size):
        t += 1
        xi = X_b_shuffled[i:i+minibatch_size]
        yi = y_shuffled[i:i+minibatch_size]
        gradients = 2/minibatch_size * xi.T.dot(xi.dot(theta) - yi)
        eta = learning_schedule(t)
        theta = theta - eta * gradients
        theta_path_mgd.append(theta)

# print(theta)
theta_path_bgd = np.array(theta_path_bgd)
theta_path_sgd = np.array(theta_path_sgd)
theta_path_mgd = np.array(theta_path_mgd)

plt.figure(figsize=(7, 4))
plt.plot(theta_path_sgd[:, 0], theta_path_sgd[:, 1], "r-s", linewidth=1, label="Stochastic")
plt.plot(theta_path_mgd[:, 0], theta_path_mgd[:, 1], "g-+", linewidth=2, label="Mini-batch")
plt.plot(theta_path_bgd[:, 0], theta_path_bgd[:, 1], "b-o", linewidth=3, label="Batch")
plt.legend(loc="upper left", fontsize=16)
plt.xlabel(r"$\theta_0$", fontsize=20)
plt.ylabel(r"$\theta_1$", fontsize=20, rotation=0)
plt.axis([2.5, 4.5, 2.3, 3.9])
save_fig("gradient_descent_paths_plot")
plt.show()
