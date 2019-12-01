# -*- coding: utf-8 -*-
# @Time    : 2018-12-23 11:09
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : ArtificalNeurlNetworks.py


import numpy as np
import os
import tensorflow as tf
from sklearn.datasets import load_iris
from sklearn.linear_model import Perceptron
from matplotlib.colors import ListedColormap
from tensorflow.contrib.layers import fully_connected
from tensorflow.examples.tutorials.mnist import input_data
from datetime import datetime

def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


import matplotlib
import matplotlib.pyplot as plt

plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIR = "."
CHAPTER_ID = "ann"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)


# iris = load_iris()
# X = iris.img[:, (2, 3)]
# y = (iris.target == 0).astype(np.int)
#
# per_clf = Perceptron(max_iter=100, tol=-np.infty, random_state=42)
# per_clf.fit(X, y)
# y_pred = per_clf.predict([[2, 0.5]])
#
# a = -per_clf.coef_[0][0] / per_clf.coef_[0][1]
# b = -per_clf.intercept_ / per_clf.coef_[0][1]
#
# axes = [0, 5, 0, 2]
#
# x0, x1 = np.meshgrid(
#     np.linspace(axes[0], axes[1], 500).reshape(-1, 1),
#     np.linspace(axes[2], axes[3], 200).reshape(-1, 1)
# )
# X_new = np.c_[x0.ravel(), x1.ravel()]
# y_predict = per_clf.predict(X_new)
# zz = y_predict.reshape(x0.shape)
#
# plt.figure(figsize=(10, 4))
# plt.plot(X[y == 0, 0], X[y == 0, 1], "bs", label="Not Iris-Setosa")
# plt.plot(X[y == 1, 0], X[y == 1, 1], "yo", label="Iris-Setosa")
#
# plt.plot([axes[0], axes[1]], [a * axes[0] + b, a * axes[1] + b], "k-", linewidth=3)
# custom_cmap = ListedColormap(['#9898ff', '#fafab0'])
#
# plt.contourf(x0, x1, zz, cmap=custom_cmap)
# plt.xlabel("Petal length", fontsize=14)
# plt.ylabel("Petal width", fontsize=14)
# plt.legend(loc="lower right", fontsize=14)
# plt.axis(axes)
#
# save_fig("perceptron_iris_plot")
# plt.show()


def logit(z):
    return 1 / (1 + np.exp(-z))


def relu(z):
    return np.maximum(0, z)


def derivative(f, z, eps=0.000001):
    return (f(z + eps) - f(z - eps))/(2 * eps)


# z = np.linspace(-5, 5, 200)
#
# plt.figure(figsize=(11, 4))
# plt.subplot(121)
# plt.plot(z, np.sign(z), "r-", linewidth=2, label="Step")
# plt.plot(z, logit(z), "g--", linewidth=2, label="Logit")
# plt.plot(z, np.tanh(z), "b-", linewidth=2, label="Tanh")
# plt.plot(z, relu(z), "m-.", linewidth=2, label="ReLU")
# plt.grid(True)
# plt.legend(loc="center right", fontsize=14)
# plt.title("Activation functions", fontsize=14)
# plt.axis([-5, 5, -1.2, 1.2])
#
# plt.subplot(122)
# plt.plot(z, derivative(np.sign, z), "r-", linewidth=2, label="Step")
# plt.plot(0, 0, "ro", markersize=5)
# plt.plot(0, 0, "rx", markersize=10)
# plt.plot(z, derivative(logit, z), "g--", linewidth=2, label="Logit")
# plt.plot(z, derivative(np.tanh, z), "b-", linewidth=2, label="Tanh")
# plt.plot(z, derivative(relu, z), "m-.", linewidth=2, label="ReLU")
# plt.grid(True)
#
# plt.title("Derivatives", fontsize=14)
# plt.axis([-5, 5, -0.2, 1.2])
#
# save_fig("activation_functions_plot")
# plt.show()

(X_train, y_train), (X_test, y_test) = keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]
#
# feature_cols = [tf.feature_column.numeric_column("X", shape=[28 * 28])]
# dnn_clf = tf.estimator.DNNClassifier(hidden_units=[300, 100], n_classes=10,
#                                      feature_columns=feature_cols)
# input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"X": X_train}, y=y_train, num_epochs=40, batch_size=50, shuffle=True
# )
# dnn_clf.train(input_fn=input_fn)
#
# test_input_fn = tf.estimator.inputs.numpy_input_fn(
#     x={"X": X_test}, y=y_test, shuffle=False
# )
# eval_results = dnn_clf.evaluate(input_fn=test_input_fn)
#
# print(eval_results)
#
# y_pred_iter = dnn_clf.predict(input_fn=test_input_fn)
# y_pred = list(y_pred_iter)
# print(y_pred[0])

# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 100
# n_outputs = 10
#
# X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
# y = tf.placeholder(tf.int64, shape=(None), name="y")
#
#
# def neuron_layer(X, n_neurons, name, activation=None):
#     with tf.name_scope(name):
#         n_inputs = int(X.get_shape()[1])
#         stddev = 2 / np.sqrt(n_inputs)
#         init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
#         W = tf.Variable(init, name="weights")
#         b = tf.Variable(tf.zeros([n_neurons]), name="biases")
#         z = tf.matmul(X, W) + b
#         if activation == "relu":
#             return tf.nn.relu(z)
#         else:
#             return z
#
#
# with tf.name_scope("dnn"):
#     hidden1 = neuron_layer(X, n_hidden1, "hidden1", activation="relu")
#     hidden2 = neuron_layer(hidden1, n_hidden2, "hidden2", activation="relu")
#     logits = neuron_layer(hidden2, n_outputs, "outputs")
#
# # with tf.name_scope("dnn"):
# #     hidden1 = fully_connected(X, n_hidden1, scope="hidden1")
# #     hidden2 = fully_connected(hidden1, hidden2, scope="hidden2")
# #     logits = fully_connected(hidden2, n_outputs, scope="outputs", activation_fn=None)
#
# # with tf.name_scope("dnn"):
# #     hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.relu)
# #     hidden2 = tf.layers.dense(hidden1, hidden2, name="hidden2", activation=tf.nn.relu)
# #     logits = tf.layers.dense(hidden2, n_outputs, name="outputs")
#
# with tf.name_scope("loss"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
#     loss = tf.reduce_mean(xentropy, name='loss')
#
# learning_rate = 0.01
#
# with tf.name_scope("train"):
#     optimizer = tf.train.GradientDescentOptimizer(learning_rate)
#     training_op = optimizer.minimize(loss)
#
# with tf.name_scope("eval"):
#     correct = tf.nn.in_top_k(logits, y, 1)
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# mnist = input_data.read_data_sets("tmp/img")
#
# n_epochs = 10
# batch_size = 50
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for iteration in range(mnist.train.num_examples // batch_size):
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         acc_train = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_test = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
#         print(epoch, "Training accuracy:", acc_train, "Test accuracy:", acc_test)
#     save_path = saver.save(sess, "./my_model_final.ckpt")
#
#
# with tf.Session() as sess:
#     saver.restore(sess, "./my_model_final.ckpt")
#     X_new_scaled = mnist.test.images[:20]
#     Z = logits.eval(feed_dict={X: X_new_scaled})
#     y_pred = np.argmax(Z, axis=1)
#
#
# print("Predicted classes:", y_pred)


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


n_inputs = 28 * 28
n_hidden1 = 300
n_hidden2 = 100
n_outputs = 10

reset_graph()

X = tf.placeholder(tf.float32, shape=(None, n_inputs), name="X")
y = tf.placeholder(tf.int32, shape=(None), name="y")

with tf.name_scope("dnn"):
    hidden1 = tf.layers.dense(X, n_hidden1, name="hidden1", activation=tf.nn.relu)
    hidden2 = tf.layers.dense(hidden1, n_hidden2, name="hidden2", activation=tf.nn.relu)
    logits = tf.layers.dense(hidden2, n_outputs, name="outputs")

with tf.name_scope("loss"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
    loss = tf.reduce_mean(xentropy, name="loss")
    loss_summary = tf.summary.scalar('log_loss', loss)

learning_rate = 0.01

with tf.name_scope("train"):
    optimizer = tf.train.GradientDescentOptimizer(learning_rate)
    training_op = optimizer.minimize(loss)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
    accuracy_summary = tf.summary.scalar('accuracy', accuracy)

init = tf.global_variables_initializer()
saver = tf.train.Saver()


def log_dir(prefix=""):
    now = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    root_logdir = "tf_logs"
    if prefix:
        prefix += "-"
    name = prefix + "run-" + now
    return "{}/{}/".format(root_logdir, name)


logdir = log_dir("mnist_dnn")

file_writer = tf.summary.FileWriter(logdir, tf.get_default_graph())

m, n = X_train.shape

n_epochs = 10001
batch_size = 50
n_batches = int(np.ceil(m / batch_size))

checkpoint_path = "tmp/my_deep_mnist_model.ckpt"
checkpoint_epoch_path = checkpoint_path + ".epoch"
final_model_path = "./my_deep_mnist_model"

best_loss = np.infty
epochs_without_progress = 0
max_epochs_without_progress = 50

with tf.Session() as sess:
    if os.path.isfile(checkpoint_epoch_path):
        with open(checkpoint_epoch_path, "rb") as f:
            start_epoch = int(f.read())
        print("Training was interrupted. Continuing at epoch", start_epoch)
        saver.restore(sess, checkpoint_path)
    else:
        start_epoch = 0
        sess.run(init)

    for epoch in range(start_epoch, n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
        accuracy_val, loss_val, accuracy_summary_str, loss_summary_str\
            = sess.run([accuracy, loss, accuracy_summary, loss_summary], feed_dict={X: X_valid, y: y_valid})
        file_writer.add_summary(accuracy_summary_str, epoch)
        file_writer.add_summary(loss_summary_str, epoch)
        if epoch % 5 == 0:
            print("Epoch:", epoch,
                  "\tValidation accuracy:{:.3f}%".format(accuracy_val * 100),
                  "\tLoss: {:.5f}".format(loss_val))
            saver.save(sess, checkpoint_path)
            with open(checkpoint_epoch_path, "wb") as f:
                f.write(b"%d" % (epoch + 1))
            if loss_val < best_loss:
                saver.save(sess, final_model_path)
                best_loss = loss_val
            else:
                epochs_without_progress += 5
                if epochs_without_progress > max_epochs_without_progress:
                    print("Early stopping")
                    break


os.remove(checkpoint_epoch_path)

with tf.Session() as sess:
    saver.restore(sess, final_model_path)
    accuracy_val = accuracy.eval(feed_dict={X: X_test, y: y_test})

print(accuracy_val)