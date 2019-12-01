# -*- coding: utf-8 -*-
# @Time    : 2018-12-21 15:50
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : RunningWithTensorflow.py

import numpy as np
import os
import tensorflow as tf
from sklearn.datasets import fetch_california_housing
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
# from tensorflow_graph_in_jupyter import show_graph
from datetime import datetime
from sklearn.datasets import make_moons
from sklearn.metrics import precision_score, recall_score
from scipy.stats import reciprocal


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12


PROJRCT_ROOT_DIR = "."
CHAPTER_ID = "tensorflow"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJRCT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


x = tf.Variable(3, name="x")
y = tf.Variable(4, name="y")
f = x*x*y + y + 2

# way1
# sess = tf.Session()
# sess.run(x.initializer)
# sess.run(y.initializer)
# result = sess.run(f)
# print(result)
# sess.close()

# way2
# with tf.Session() as sess:
#     x.initializer.run()
#     y.initializer.run()
#     result = f.eval()
# print(result)

# way3
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     init.run()
#     result = f.eval()
#     print(result)

# way4
# init = tf.global_variables_initializer()
# sess = tf.InteractiveSession()
# init.run()
# result = f.eval()
# print(result)
# sess.close()

# x1 = tf.Variable(1)
# print(x1.graph is tf.get_default_graph())

# graph = tf.Graph()
# with graph.as_default():
#     x2 = tf.Variable(2)
#
# print(x2.graph is graph)
#
# print(x2.graph is tf.get_default_graph())

# w = tf.constant(3)
# x = w + 2
# y = x + 5
# z = x * 3
#
# with tf.Session() as sess:
#     print(y.eval())
#     print(z.eval())

# 共用一个Graph
# with tf.Session() as sess:
#     y_val, z_val = sess.run([y, z])
#     print(y_val)
#     print(z_val)

housing = fetch_california_housing()
m, n = housing.data.shape
housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.data]

# 正态方程
# tensorflow
# X = tf.constant(housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# XT = tf.transpose(X)
# theta = tf.matmul(tf.matmul(tf.matrix_inverse(tf.matmul(XT, X)), XT), y)
# with tf.Session() as sess:
#     theta_value = theta.eval()
# print(theta_value)

# numpy
# X = housing_data_plus_bias
# y = housing.target.reshape(-1, 1)
# theta_numpy = np.linalg.inv(X.T.dot(X)).dot(X.T).dot(y)
# print(theta_numpy)

# sklearn
# lin_reg = LinearRegression()
# lin_reg.fit(housing_data_plus_bias, housing.target.reshape(-1, 1))
#
# print(np.r_[lin_reg.intercept_.reshape(-1, 1), lin_reg.coef_.T])

# scaler = StandardScaler()
# scaled_housing_data = scaler.fit_transform(housing.img)
# scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
#
# reset_graph()
#
# n_epochs = 1000
# learning_rate = 0.01
#
# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
# y_pred = tf.matmul(X, theta, name="predictions")
# error = y_pred - y
# mse = tf.reduce_mean(tf.square(error), name="mse")
# gradients = 2/m*tf.matmul(tf.transpose(X), error) # gradients = tf.gradients(mse, [theta])[0]
# training_op = tf.assign(theta, theta - learning_rate * gradients)
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(n_epochs):
#         if epoch % 100 == 0:
#             print("Epoch", epoch, "^MSE = ", mse.eval())
#         sess.run(training_op)
#     best_theta = theta.eval()
#
# print(best_theta)

# 梯度下降优化器
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# 动量优化器
# optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=0.9)
# training_op = optimizer.minimize(mse)
#
# init = tf.global_variables_initializer()
# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(n_epochs):
#         if epoch % 100 == 0:
#             print("Epoch", epoch, "MSE = ", mse.eval())
#         sess.run(training_op)
#     best_theta = theta.eval()
#
# print("Best theta:")
# print(best_theta)

# A = tf.placeholder(tf.float32, shape=(None, 3))
# B = A + 5
# with tf.Session() as sess:
#     B_val_1 = B.eval(feed_dict={A: [[1, 2, 3]]})
#     B_val_2 = B.eval(feed_dict={A: [[4, 5, 6], [7, 8, 9]]})
#
# print(B_val_1)
# print(B_val_2)

# mini-batch
# learning_rate = 0.01
#
# reset_graph()
#
# X = tf.placeholder(tf.float32, shape=(None, n+1), name="X")
# y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
#
# theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
# y_pred = tf.matmul(X, theta, name="predictions")
# error = y_pred - y
# mse = tf.reduce_mean(tf.square(error), name="mse")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)
#
# init = tf.global_variables_initializer()
#
# n_epochs = 10
# batch_size = 100
# n_batches = int(np.ceil(m/batch_size))


# def fetch_batch(epoch, batch_index, batch_size):
#     np.random.seed(epoch * n_batches + batch_index)
#     indices = np.random.randint(m, size=batch_size)
#     X_batch = scaled_housing_data_plus_bias[indices]
#     y_batch = housing.target.reshape(-1, 1)[indices]
#     return X_batch, y_batch


# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(n_epochs):
#         for batch_index in range(n_batches):
#             X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#
#     best_theta = theta.eval()
#
# print(best_theta)

# reset_graph()
#
# n_epochs = 1000
# learning_rate = 0.01
#
# X = tf.constant(scaled_housing_data_plus_bias, dtype=tf.float32, name="X")
# y = tf.constant(housing.target.reshape(-1, 1), dtype=tf.float32, name="y")
# theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
# y_pred = tf.matmul(X, theta, name="predictions")
# error = y_pred - y
# mse = tf.reduce_mean(tf.square(error), name="mse")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(n_epochs):
#         if epoch % 100 == 0:
#             print("Epoch", epoch, "MSE = ", mse.eval())
#             save_path = saver.save(sess, "tmp/my_model.ckpt")
#         sess.run(training_op)
#     best_theta = theta.eval()
#     save_path = saver.save(sess, "tmp/my_model_final.ckpt")
#
# print(best_theta)
#
# with tf.Session() as sess:
#     saver.restore(sess, "tmp/my_model_final.ckpt")
#     best_theta_restored = theta.eval()
#
# print(np.allclose(best_theta, best_theta_restored))
#
# saver = tf.train.Saver({"weights": theta})
#
# reset_graph()
#
# saver = tf.train.import_meta_graph("tmp/my_model_final.ckpt.meta")
# theta = tf.get_default_graph().get_tensor_by_name("theta:0")
#
# with tf.Session(    ) as sess:
#     saver.restore(sess, "tmp/my_model_final.ckpt")
#     best_theta_restored = theta.eval()
#
# print(np.allclose(best_theta, best_theta_restored))
#
# tf.summary.FileWriter("logs", tf.get_default_graph()).close()

# show_graph(tf.get_default_graph())

# reset_graph()
#
# housing = fetch_california_housing()
# m, n = housing.img.shape
# housing_data_plus_bias = np.c_[np.ones((m, 1)), housing.img]
# scaler = StandardScaler()
# scaled_housing_data = scaler.fit_transform(housing.img)
# scaled_housing_data_plus_bias = np.c_[np.ones((m, 1)), scaled_housing_data]
#
#
# now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_log_dirs = "tf_logs"
# log_dir = "{}/run-{}/".format(root_log_dirs, now)
#
# n_epochs = 1000
# learning_rate = 0.01
#
# X = tf.placeholder(tf.float32, shape=(None, n + 1), name="X")
# y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# theta = tf.Variable(tf.random_uniform([n+1, 1], -1.0, 1.0, seed=42), name="theta")
# y_pred = tf.matmul(X, theta, name="predictions")
# with tf.name_scope("loss") as scope:
#     error = y_pred - y
#     mse = tf.reduce_mean(tf.square(error), name="mse")
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(mse)
#
# init = tf.global_variables_initializer()
#
# mse_summary = tf.summary.scalar('MSE', mse)
# file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
#
# n_epochs = 10
# batch_size = 100
# n_batches = int(np.ceil(m / batch_size))
#
#
# def fetch_batch(epoch, batch_index, batch_size):
#     np.random.seed(epoch * n_batches + batch_index)
#     indices = np.random.randint(m, size=batch_size)
#     X_batch = scaled_housing_data_plus_bias[indices]
#     y_batch = housing.target.reshape(-1, 1)[indices]
#     return X_batch, y_batch
#
#
# with tf.Session() as sess:
#     sess.run(init)
#
#     for epoch in range(n_epochs):
#         for batch_index in range(n_epochs):
#             X_batch, y_batch = fetch_batch(epoch, batch_index, batch_size)
#             if batch_index % 10 == 0:
#                 summary_str = mse_summary.eval(feed_dict={X: X_batch, y: y_batch})
#                 step = epoch * n_batches + batch_index
#                 file_writer.add_summary(summary_str, step)
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#     best_theta = theta.eval()
#
# file_writer.close()
# print(best_theta)

# reset_graph()
#
# n_features = 3
# X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
# w1 = tf.Variable(tf.random_normal((n_features, 1)), name="weights1")
# w2 = tf.Variable(tf.random_normal((n_features, 1)), name="weights2")
# b1 = tf.Variable(0.0, name="bias1")
# b2 = tf.Variable(0.0, name="bias2")
# z1 = tf.add(tf.matmul(X, w1), b1, name="z1")
# z2 = tf.add(tf.matmul(X, w2), b2, name="z2")
# relu1 = tf.maximum(z1, 0., name="relu1")
# relu2 = tf.maximum(z2, 0., name="relu2")
# output = tf.add(relu1, relu2, name="output")
#
# now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_log_dirs = "tf_logs"
# log_dir = "{}/run-{}/".format(root_log_dirs, now)
# file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
# file_writer.close()

# reset_graph()
#
#
# def relu(X):
#     threshold = tf.get_variable("threshold", shape=(),
#                                 initializer=tf.constant_initializer(0.0))
#     w_shape = (int(X.get_shape()[1]), 1)
#     w = tf.Variable(tf.random_normal(w_shape), name="weights")
#     b = tf.Variable(0.0, name="bias")
#     z = tf.add(tf.matmul(X, w), b, name="z")
#     return tf.maximum(z, threshold, name="relu")
#
#
# n_features = 3
# X = tf.placeholder(tf.float32, shape=(None, n_features), name="X")
# relus = []
# for relu_index in range(5):
#     with tf.variable_scope("relu", reuse=(relu_index >= 1)) as scope:
#         relus.append(relu(X))
# output = tf.add_n(relus, name="output")
#
# now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
# root_log_dirs = "tf_logs"
# log_dir = "{}/run-{}/".format(root_log_dirs, now)
# file_writer = tf.summary.FileWriter(log_dir, tf.get_default_graph())
# file_writer.close()

# reset_graph()
#
# with tf.variable_scope("my_scope"):
#     x0 = tf.get_variable("x", shape=(), initializer=tf.constant_initializer(0.))
#     x1 = tf.Variable(0, name="x")
#     x2 = tf.Variable(0, name="x")
#
# with tf.variable_scope("my_scope", reuse=True):
#     x3 = tf.get_variable("x")
#     x4 = tf.Variable(0., name="x")
#
# with tf.variable_scope("", default_name="", reuse=True):
#     x5 = tf.get_variable("my_scope/x")

m = 1000
X_moons, y_moons = make_moons(m, noise=0.1, random_state=42)

# plt.plot(X_moons[y_moons == 1, 0], X_moons[y_moons == 1, 1], 'go', label="Positive")
# plt.plot(X_moons[y_moons == 0, 0], X_moons[y_moons == 0, 1], 'r^', label="Negative")
# plt.legend()
# plt.show()

X_moons_with_bias = np.c_[np.ones((m, 1)), X_moons]
y_moons_column_vector = y_moons.reshape(-1, 1)

test_ratio = 0.2
test_size = int(m * test_ratio)
X_train = X_moons_with_bias[:test_size]
X_test = X_moons_with_bias[-test_size:]
y_train = y_moons_column_vector[:-test_size]
y_test = y_moons_column_vector[-test_size:]


def random_batch(X_train, y_train, batch_size):
    rnd_indice = np.random.randint(0, len(X_train), batch_size)
    X_batch = X_train[rnd_indice]
    y_batch = y_train[rnd_indice]
    return X_batch, y_batch


# reset_graph()
#
# n_inputs = 2
#
# X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
# y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
# theta = tf.Variable(tf.random_uniform([n_inputs + 1, 1], -1.0, 1.0, seed=42), name="theta")
# logits = tf.matmul(X, theta, name="logits")
# y_proba = 1 / (1 + tf.exp(-logits))
# # y_proba = tf.sigmoid(logits)
#
# epsilon = 1e-7
# loss = - tf.reduce_mean(y * tf.log(y_proba + epsilon) + (1 - y)*tf.log(1 - y_proba + epsilon))
# loss = tf.losses.log_loss(y, y_proba)
#
# learning_rate = 0.01
# optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
# training_op = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
#
# n_epochs = 1000
# batch_size = 50
# n_batches = int(np.ceil(m/batch_size))
#
# with tf.Session() as sess:
#     sess.run(init)
#     for epoch in range(n_epochs):
#         for batch_index in range(n_batches):
#             X_batch, y_batch = random_batch(X_train, y_train, batch_size)
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#             loss_val = loss.eval({X: X_test, y: y_test})
#         if epoch % 100 == 0:
#             print("Epoch:", epoch, "\tLoss:", loss_val)
#
#     y_proba_val = y_proba.eval(feed_dict={X: X_test, y: y_test})
#
# y_pred = (y_proba_val >= 0.5)
# print(y_pred[:5])

# print(precision_score(y_test, y_pred))
# print(recall_score(y_test, y_pred))

# y_pred_idx = y_pred.reshape(-1)
# plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], "go", label="Positive")
# plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], "r^", label="Negative")
# plt.legend()
# plt.show()

reset_graph()

X_train_enhanced = np.c_[X_train, np.square(X_train[:, 1]), np.square(X_train[:, 2]),
                         X_train[:, 1] ** 3, X_train[:, 2] ** 3]
X_test_enhanced = np.c_[X_test, np.square(X_test[:, 1]), np.square(X_test[:, 1]),
                        X_test[:, 1] ** 3, X_test[:, 2] ** 3]


def logistic_regression(X, y, initializer=None, seed=42, learning_rate=0.01):
    n_inputs_including_bias = int(X.get_shape()[1])
    with tf.name_scope("logistic_regression"):
        with tf.name_scope("model"):
            if initializer is None:
                initializer = tf.random_uniform([n_inputs_including_bias, 1], -1.0, 1.0, seed=seed)
            theta = tf.Variable(initializer, name="theta")
            logits = tf.matmul(X, theta, name="logits")
            y_proba = tf.sigmoid(logits)
        with tf.name_scope("train"):
            loss = tf.losses.log_loss(y, y_proba, scope="loss")
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=learning_rate)
            training_op = optimizer.minimize(loss)
            loss_summary = tf.summary.scalar('log_loss', loss)
        with tf.name_scope("init"):
            init = tf.global_variables_initializer()
        with tf.name_scope("saver"):
            saver = tf.train.Saver()
    return y_proba, loss, training_op, loss_summary, init, saver


def log_dir(prefix=''):
    now = datetime.utcnow().strftime("%Y%M%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


# n_inputs = 2 + 4
# logdir = log_dir("log_reg")
#
# X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
# y = tf.placeholder(tf.float32, shape=(None, 1), name="y")
#
# y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(X, y)
#
# file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())
#
# n_epochs = 10001
# batch_size = 50
# n_batches = int(np.ceil(m / batch_size))
#
# checkpoint_path = "tmp/my_logreg_model.ckpt"
# checkpoint_epoch_path = checkpoint_path + ".epoch"
# final_model_path = "./my_log_reg_model"
#
# with tf.Session() as sess:
#     if os.path.isfile(checkpoint_epoch_path):
#         with open(checkpoint_epoch_path, "rb") as f:
#             start_epoch = int(f.read())
#         print("Training was interrupted. Continuing at epoch", start_epoch)
#         saver.restore(sess, checkpoint_path)
#     else:
#         start_epoch = 0
#         sess.run(init)
#
#     for epoch in range(start_epoch, n_epochs):
#         for batch_index in range(n_batches):
#             X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
#         file_writer.add_summary(summary_str, epoch)
#         if epoch % 500 == 0:
#             print("Epoch:", epoch, "\tLoss", loss_val)
#             saver.save(sess, checkpoint_path)
#             with open(checkpoint_epoch_path, "wb") as f:
#                 f.write(b"%d" % (epoch + 1))
#     saver.save(sess, final_model_path)
#     y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
#     os.remove(checkpoint_epoch_path)
#
# y_pred = (y_proba_val >= 0.5)
# print(precision_score(y_test, y_pred))
# print(recall_score(y_test, y_pred))
#
# y_pred_idx = y_pred.reshape(-1)
# plt.plot(X_test[y_pred_idx, 1], X_test[y_pred_idx, 2], "go", label="Positive")
# plt.plot(X_test[~y_pred_idx, 1], X_test[~y_pred_idx, 2], "r^", label="Negative")
# plt.legend()
# plt.show()

n_search_iterations = 10

for search_iteration in range(n_search_iterations):
    batch_size = np.random.randint(1, 100)
    learning_rate = reciprocal(0.0001, 0.1).rvs(random_state=search_iteration)

    n_inputs = 2 + 4
    logdir = log_dir("log_reg")

    print("Iteration", search_iteration)
    print("logdir:", logdir)
    print("batchsize:", batch_size)
    print("learning_rate:", learning_rate)
    print("training:", end="")

    reset_graph()

    X = tf.placeholder(tf.float32, shape=(None, n_inputs + 1), name="X")
    y = tf.placeholder(tf.float32, shape=(None, 1), name="y")

    y_proba, loss, training_op, loss_summary, init, saver = logistic_regression(X, y, learning_rate=learning_rate)

    file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

    n_epochs = 10001
    n_batches = int(np.ceil(m / batch_size))

    final_model_path = "./my_logreg_model_%d" % search_iteration

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(n_epochs):
            for batch_index in range(n_batches):
                X_batch, y_batch = random_batch(X_train_enhanced, y_train, batch_size)
                sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
            loss_val, summary_str = sess.run([loss, loss_summary], feed_dict={X: X_test_enhanced, y: y_test})
            file_writer.add_summary(summary_str, epoch)
            if epoch % 500 == 0:
                print(".", end="")
        saver.save(sess, final_model_path)

        print()
        y_proba_val = y_proba.eval(feed_dict={X: X_test_enhanced, y: y_test})
        y_pred = (y_proba_val > 0.5)
        print("precision:", precision_score(y_test, y_pred))
        print("recall:", recall_score(y_test, y_pred))


