# -*- coding: utf-8 -*-
# @Time    : 2019-01-21 17:19
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : AutoEncoders.py

import tensorflow as tf
import numpy as np
import os
import sys
import numpy.random as rnd
from sklearn.preprocessing import StandardScaler
from tensorflow.examples.tutorials.mnist import input_data
from functools import partial


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
CHAPTER_ID = "autoencoders"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)


def plot_image(image, shape=[28, 28]):
    plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
    plt.axis("off")


def plot_multiple_images(images, n_rows, n_cols, pad=2):
    images = images - images.min()
    w, h = images.shape[1:]
    image = np.zeros(((w + pad) * n_rows + pad, (h + pad) * n_cols + pad))
    for y in range(n_rows):
        for x in range(n_cols):
            image[(y * (h + pad) + pad):(y * (h + pad) + pad + h), (x * (w + pad) + pad):(x * (w + pad) + pad + w)] =\
                images[y * n_cols + x]
    plt.imshow(image, cmap="Greys", interpolation="nearest")
    plt.axis("off")


# PCA with a linear Autoencoder
# rnd.seed(42)
# m = 200
# w1, w2 = 0.1, 0.3
# noise = 0.1
#
# angles = rnd.rand(m) * 3 * np.pi / 2 - 0.5
# img = np.empty((m, 3))
# img[:, 0] = np.cos(angles) + np.sin(angles)/2 + noise * rnd.randn(m)/2
# img[:, 1] = np.sin(angles) * 0.7 + noise * rnd.rand(m) / 2
# img[:, 2] = img[:, 0] * w1 + img[:, 1] + noise * rnd.randn(m)
#
# scaler = StandardScaler()
# X_train = scaler.fit_transform(img[:100])
# X_test = scaler.fit_transform(img[100:])
#
# reset_graph()
#
# n_inputs = 3
# n_hidden = 2
# n_outputs = n_inputs
#
# learning_rate = 0.01
#
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# hidden = tf.layers.dense(X, n_hidden)
# outputs = tf.layers.dense(hidden, n_outputs)
#
# reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
#
# optimizer = tf.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(reconstruction_loss)
#
# init = tf.global_variables_initializer()
#
# n_iterations = 1000
# codings = hidden
#
# with tf.Session() as sess:
#     init.run()
#     for iteraiton in range(n_iterations):
#         training_op.run(feed_dict={X: X_test})
#     codings_val = codings.eval(feed_dict={X: X_test})
#
# fig = plt.figure(figsize=(4, 3))
# plt.plot(codings_val[:, 0], codings_val[:, 1], "b.")
# plt.xlabel("$z_1$", fontsize=18)
# plt.ylabel("$z_2$", fontsize=18, rotation=0)
# save_fig("linear_autoencoder_pca_plot")
# plt.show()

# Stacked Autoencoders
mnist = input_data.read_data_sets("tmp/img/")

# Train all layers at once
# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 150
# n_hidden3 = n_hidden1
# n_outputs = n_inputs
#
# learning_rate = 0.01
# l2_reg = 0.0001
#
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# he_init = tf.contrib.layers.variance_scaling_initializer()
# # he_init = lambda shape, dtype=tf.float32: tf.truncated_normal(shape, 0, stddev=np.sqrt(2/shape[0]))
# l2_regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
# my_dense_layer = partial(tf.layers.dense,
#                          activation=tf.nn.elu,
#                          kernel_initializer=he_init,
#                          kernel_regularizer=l2_regularizer)
# hidden1 = my_dense_layer(X, n_hidden1)
# hidden2 = my_dense_layer(hidden1, n_hidden2)
# hidden3 = my_dense_layer(hidden2, n_hidden3)
# outputs = my_dense_layer(hidden3, n_outputs, activation=None)
#
# reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
# reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
# loss = tf.add_n([reconstruction_loss] + reg_losses)
#
# optimizer = tf.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# # n_epochs = 5
# # batch_size = 150
# #
# # with tf.Session() as sess:
# #     init.run()
# #     for epoch in range(n_epochs):
# #         n_batches = mnist.train.num_examples // batch_size
# #         for iteration in range(n_batches):
# #             print("\r{}%".format(100 * iteration // n_batches), end=" ")
# #             sys.stdout.flush()
# #             X_batch, y_batch = mnist.train.next_batch(batch_size)
# #             sess.run(training_op, feed_dict={X: X_batch})
# #         loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
# #         print("\r{}".format(epoch), "Train_MSE:", loss_train)
# #         saver.save(sess, "./my_model_all_layers.ckpt")


def show_reconstructed_digits(X, outputs, model_path=None, n_test_digits=2):
    with tf.Session() as sess:
        if model_path:
            saver.restore(sess, model_path)
        X_test = mnist.test.images[:n_test_digits]
        outputs_val = outputs.eval(feed_dict={X: X_test})
    fig = plt.figure(figsize=(8, 3 * n_test_digits))
    for digit_index in range(n_test_digits):
        plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
        plot_image(X_test[digit_index])
        plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
        plot_image(outputs_val[digit_index])
    plt.show()


# # show_reconstructed_digits(X, outputs, "./my_model_all_layers.ckpt")
# # save_fig("reconstruction_plot")

# Tying weights
# reset_graph()
#
# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 150
# n_hidden3 = n_hidden1
# n_outputs = n_inputs
#
# learning_rate = 0.01
# l2_reg = 0.0005
#
# activation = tf.nn.elu
# regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
# initializer = tf.contrib.layers.variance_scaling_initializer()
#
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
#
# weights1_init = initializer([n_inputs, n_hidden1])
# weights2_init = initializer([n_hidden1, n_hidden2])
#
# weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
# weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
# weights3 = tf.transpose(weights2, name="weights3")
# weights4 = tf.transpose(weights1, name="weights4")
#
# biases1 = tf.Variable(tf.zeros(n_hidden1), name='biases1')
# biases2 = tf.Variable(tf.zeros(n_hidden2), name='biases2')
# biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
# biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")
#
# hidden1 = activation(tf.matmul(X, weights1) + biases1)
# hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
# hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
# outputs = tf.matmul(hidden3, weights4) + biases4
#
# reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
# reg_loss = regularizer(weights1) + regularizer(weights2)
# loss = reconstruction_loss + reg_loss
#
# optimizer = tf.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
#
# saver = tf.train.Saver()
#
# n_epochs = 4
# batch_size = 150
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         n_batches = mnist.train.num_examples // batch_size
#         for iteration in range(n_batches):
#             print("\r{}%".format(100 * iteration // n_batches), end=" ")
#             sys.stdout.flush()
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             sess.run(training_op, feed_dict={X: X_batch})
#         loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
#         print("\r{}".format(epoch), "Train MSE:", loss_train)
#         saver.save(sess, "./my_model_tying_weights.ckpt")
#
# show_reconstructed_digits(X, outputs, "./my_model_tying_weights.ckpt")

# Training one Autoencoder at a time in multiple graphs
# reset_graph()
#
#
# def train_autoencoder(X_train, n_neurons, n_epochs, batch_size,
#                       learning_rate=0.01, l2_reg=0.0005, seed=42,
#                       hidden_activation=tf.nn.elu,
#                       output_activaiton=tf.nn.elu):
#     graph = tf.Graph()
#     with graph.as_default():
#         tf.set_random_seed(seed)
#         n_inputs = X_train.shape[1]
#         X = tf.placeholder(tf.float32, shape=[None, n_inputs])
#         my_dense_layer = partial(
#             tf.layers.dense,
#             kernel_initializer=tf.contrib.layers.variance_scaling_initializer(),
#             kernel_regularizer=tf.contrib.layers.l2_regularizer(l2_reg)
#         )
#         hidden = my_dense_layer(X, n_neurons, activation=hidden_activation, name="hidden")
#         outputs = my_dense_layer(hidden, n_inputs, activation=output_activaiton, name="outputs")
#
#         reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
#
#         reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
#         loss = tf.add_n([reconstruction_loss] + reg_losses)
#
#         optimizer = tf.train.AdamOptimizer(learning_rate)
#         trainging_op = optimizer.minimize(loss)
#
#         init = tf.global_variables_initializer()
#
#     with tf.Session(graph=graph) as sess:
#         init.run()
#         for epoch in range(n_epochs):
#             n_batches = len(X_train) // batch_size
#             for iteration in range(n_batches):
#                 print("\r{}%".format(100 * iteration // n_batches), end=" ")
#                 sys.stdout.flush()
#                 indices = rnd.permutation(len(X_train))[:batch_size]
#                 X_batch = X_train[indices]
#                 sess.run(trainging_op, feed_dict={X: X_batch})
#             loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
#             print("\r{}".format(epoch), "Train MSE:", loss_train)
#         params = dict([(var.name, var.eval()) for var in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES)])
#         hidden_val = hidden.eval(feed_dict={X: X_train})
#         return hidden_val, params['hidden/kernel:0'], params["hidden/bias:0"],\
#                params["outputs/kernel:0"], params["outputs/bias:0"]
#
#
# hidden_output, W1, b1, W4, b4 = train_autoencoder(mnist.train.images, n_neurons=400, n_epochs=4, batch_size=150,
#                                                   output_activaiton=None)
# _, W2, b2, W3, b3 = train_autoencoder(hidden_output, n_neurons=150, n_epochs=4, batch_size=150)
#
# reset_graph()
#
# n_inputs = 28 * 28
#
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# hidden1 = tf.nn.elu(tf.matmul(X, W1) + b1)
# hidden2 = tf.nn.elu(tf.matmul(hidden1, W2) + b2)
# hidden3 = tf.nn.elu(tf.matmul(hidden2, W3) + b3)
# outputs = tf.matmul(hidden3, W4) + b4
#
# show_reconstructed_digits(X, outputs)

# Training one Autoencoder at a time in a single graph
# reset_graph()
#
# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 150
# n_hidden3 = n_hidden1
# n_outputs = n_inputs
#
# learning_rate = 0.01
# l2_reg = 0.0001
#
# activation = tf.nn.elu
# regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
# initializer = tf.contrib.layers.variance_scaling_initializer()
#
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
#
# weights1_init = initializer([n_inputs, n_hidden1])
# weights2_init = initializer([n_hidden1, n_hidden2])
# weights3_init = initializer([n_hidden2, n_hidden3])
# weights4_init = initializer([n_hidden3, n_outputs])
#
# weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
# weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
# weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
# weights4 = tf.Variable(weights4_init, dtype=tf.float32, name="weights4")
#
# biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
# biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
# biases3 = tf.Variable(tf.zeros(n_hidden3), name="biases3")
# biases4 = tf.Variable(tf.zeros(n_outputs), name="biases4")
#
# hidden1 = activation(tf.matmul(X, weights1) + biases1)
# hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
# hidden3 = activation(tf.matmul(hidden2, weights3) + biases3)
# outputs = tf.matmul(hidden3, weights4) + biases4
#
# reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
#
# optimizer = tf.train.AdamOptimizer(learning_rate)
#
# with tf.name_scope("phase1"):
#     phase1_outputs = tf.matmul(hidden1, weights4) + biases4
#     phase1_reconstruction_loss = tf.reduce_mean(tf.square(phase1_outputs - X))
#     phase1_reg_loss = regularizer(weights1) + regularizer(weights4)
#     phase1_loss = phase1_reconstruction_loss + phase1_reg_loss
#     phase1_training_op = optimizer.minimize(phase1_loss)
#
# with tf.name_scope("phase2"):
#     phase2_reconstruction_loss = tf.reduce_mean(tf.square(hidden3 - hidden1))
#     phase2_reg_loss = regularizer(weights2) + regularizer(weights3)
#     phase2_loss = phase2_reconstruction_loss + phase2_reg_loss
#     train_vars = [weights2, biases2, weights3, biases3]
#     phase2_training_op = optimizer.minimize(phase2_loss, var_list=train_vars)
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# training_ops = [phase1_training_op, phase2_training_op]
# reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss]
# n_epochs = [4, 4]
# batch_sizes = [150, 150]

# with tf.Session() as sess:
#     init.run()
#     for phase in range(2):
#         print("Training phase #{}".format(phase + 1))
#         for epoch in range(n_epochs[phase]):
#             n_batches = mnist.train.num_examples // batch_sizes[phase]
#             for iteration in range(n_batches):
#                 print("\r{}%".format(100 * iteration // n_batches), end=" ")
#                 sys.stdout.flush()
#                 X_batch, y_batch = mnist.train.next_batch(batch_sizes[phase])
#                 sess.run(training_ops[phase], feed_dict={X: X_batch})
#             loss_train = reconstruction_losses[phase].eval(feed_dict={X: X_batch})
#             print("\r{}".format(epoch), "Train MSE:", loss_train)
#             saver.save(sess, "./my_model_one_at_a_time.ckpt")
#     loss_test = reconstruction_loss.eval(feed_dict={X: mnist.test.images})
#     print("Test MSE:", loss_test)

# cache the frozen layer outputs
# training_ops = [phase1_training_op, phase2_training_op]
# reconstruction_losses = [phase1_reconstruction_loss, phase2_reconstruction_loss]
# n_epochs = [4, 4]
# batch_sizes = [150, 150]
#
# with tf.Session() as sess:
#     init.run()
#     for phase in range(2):
#         print("Training phase #{}".format(phase + 1))
#         if phase == 1:
#             hidden1_cache = hidden1.eval(feed_dict={X: mnist.train.images})
#         for epoch in range(n_epochs[phase]):
#             n_batches = mnist.train.num_examples // batch_sizes[phase]
#             for iteration in range(n_batches):
#                 print("\r{}%".format(100 * iteration // n_batches), end=" ")
#                 sys.stdout.flush()
#                 if phase == 1:
#                     indices = rnd.permutation(mnist.train.num_examples)
#                     hidden1_batch = hidden1_cache[indices[:batch_sizes[phase]]]
#                     feed_dict = {hidden1: hidden1_batch}
#                     sess.run(training_ops[phase], feed_dict=feed_dict)
#                 else:
#                     X_batch, y_batch = mnist.train.next_batch(batch_sizes[phase])
#                     feed_dict = {X: X_batch}
#                     sess.run(training_ops[phase], feed_dict=feed_dict)
#             loss_train = reconstruction_losses[phase].eval(feed_dict=feed_dict)
#             print("\r{}".format(epoch), "Train MSE:", loss_train)
#             saver.save(sess, "./my_model_cache_frozen.ckpt")
#     loss_test = reconstruction_loss.eval(feed_dict={X: mnist.test.images})
#     print("Test MSE:", loss_test)

# Visualizing the Reconstructions
# n_test_digits = 2
# X_test = mnist.test.images[:n_test_digits]
#
# with tf.Session() as sess:
#     saver.restore(sess, "./my_model_one_at_a_time.ckpt")
#     outputs_val = outputs.eval(feed_dict={X: X_test})
#
#
# def plot_image(image, shape=[28, 28]):
#     plt.imshow(image.reshape(shape), cmap="Greys", interpolation="nearest")
#     plt.axis("off")
#
#
# for digit_index in range(n_test_digits):
#     plt.subplot(n_test_digits, 2, digit_index * 2 + 1)
#     plot_image(X_test[digit_index])
#     plt.subplot(n_test_digits, 2, digit_index * 2 + 2)
#     plot_image(outputs_val[digit_index])
# plt.show()

# Visualizing the extracted features
# with tf.Session() as sess:
#     saver.restore(sess, "./my_model_one_at_a_time.ckpt")
#     weights1_val = weights1.eval()
#
# for i in range(5):
#     plt.subplot(1, 5, i+1)
#     plot_image(weights1_val.T[i])
#
# save_fig("extracted_features_plot")
# plt.show()

# Unsupervised pretraining
# reset_graph()
#
# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 150
# n_outputs = 10
#
# learning_rate = 0.01
# l2_reg = 0.0005
#
# activation = tf.nn.elu
# regularizer = tf.contrib.layers.l2_regularizer(l2_reg)
# initializer = tf.contrib.layers.variance_scaling_initializer()
#
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# y = tf.placeholder(tf.int32, shape=[None])
#
# weights1_init = initializer([n_inputs, n_hidden1])
# weights2_init = initializer([n_hidden1, n_hidden2])
# weights3_init = initializer([n_hidden2, n_outputs])
#
# weights1 = tf.Variable(weights1_init, dtype=tf.float32, name="weights1")
# weights2 = tf.Variable(weights2_init, dtype=tf.float32, name="weights2")
# weights3 = tf.Variable(weights3_init, dtype=tf.float32, name="weights3")
#
# biases1 = tf.Variable(tf.zeros(n_hidden1), name="biases1")
# biases2 = tf.Variable(tf.zeros(n_hidden2), name="biases2")
# biases3 = tf.Variable(tf.zeros(n_outputs), name="biases3")
#
# hidden1 = activation(tf.matmul(X, weights1) + biases1)
# hidden2 = activation(tf.matmul(hidden1, weights2) + biases2)
# logits = tf.matmul(hidden2, weights3) + biases3
#
# cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=y, logits=logits)
# reg_loss = regularizer(weights1) + regularizer(weights2) + regularizer(weights3)
# loss = cross_entropy + reg_loss
# optimizer = tf.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(loss)
#
# correct = tf.nn.in_top_k(logits, y, 1)
# accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# init = tf.global_variables_initializer()
# pretrain_saver = tf.train.Saver([weights1, weights2, biases1, biases2])
# saver = tf.train.Saver()
#
# # n_epochs = 4
# # batch_size = 150
# # n_labeled_instances = 20000
# #
# # with tf.Session() as sess:
# #     init.run()
# #     for epoch in  range(n_epochs):
# #         n_batches = n_labeled_instances // batch_size
# #         for iteration in range(n_batches):
# #             print("\r{}%".format(100 * iteration // n_batches), end=" ")
# #             sys.stdout.flush()
# #             indices = rnd.permutation(n_labeled_instances)[:batch_size]
# #             X_batch, y_batch = mnist.train.images[indices], mnist.train.labels[indices]
# #             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
# #         accuracy_val = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
# #         print("\r{}".format(epoch), "Train accuracy:", accuracy_val, end=" ")
# #         saver.save(sess, "./my_model_supervised.ckpt")
# #         accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
# #         print("Test accuracy:", accuracy_val)
#
# n_epochs = 4
# batch_size = 150
# n_labeled_instance = 20000
#
# with tf.Session() as sess:
#     init.run()
#     pretrain_saver.restore(sess, "./my_model_cache_frozen.ckpt")
#     for epoch in range(n_epochs):
#         n_batches = n_labeled_instance // batch_size
#         for iteration in range(n_batches):
#             print("\r{}%".format(100 * iteration // n_batches), end=" ")
#             sys.stdout.flush()
#             indices = rnd.permutation(n_labeled_instance)[:batch_size]
#             X_batch, y_batch = mnist.train.images[indices], mnist.train.labels[indices]
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         accuracy_val = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         print("\r{}".format(epoch), "Train accuracy:", accuracy_val, end="\t")
#         saver.save(sess, "./my_model_supervised_pretrained.ckpt")
#         accuracy_val = accuracy.eval(feed_dict={X: mnist.test.images, y: mnist.test.labels})
#         print("Test accuracy:", accuracy_val)

# Stacked denoising Autoencoder(Gaussian noise)
# reset_graph()
#
# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 150
# n_hidden3 = n_hidden1
# n_outputs = n_inputs
#
# learning_rate = 0.01
# noise_level = 1.0
#
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# X_noisy = X + noise_level * tf.random_normal(tf.shape(X))
#
# hidden1 = tf.layers.dense(X_noisy, n_hidden1, activation=tf.nn.relu, name="hidden1")
# hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu, name="hidden2")
# hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu, name="hidden3")
# outputs = tf.layers.dense(hidden3, n_outputs, name="outputs")
#
# reconsturction_loss = tf.reduce_mean(tf.square(outputs - X))
#
# optimizer = tf.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(reconsturction_loss)
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# n_epochs = 10
# batch_size = 150
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         n_batches = mnist.train.num_examples // batch_size
#         for iteration in range(n_batches):
#             print("\r{}%".format(100 * iteration // n_batches), end=" ")
#             sys.stdout.flush()
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             sess.run(training_op, feed_dict={X: X_batch})
#         loss_train = reconsturction_loss.eval(feed_dict={X: X_batch})
#         print("\r{}".format(epoch), "Train MSE:", loss_train)
#         saver.save(sess, "./my_model_stacked_denoising_guassian.ckpt")

# Stacked denoising Autoencoder(using dropout)
# reset_graph()
#
# n_inputs = 28 * 28
# n_hidden1 = 300
# n_hidden2 = 150
# n_hidden3 = n_hidden1
# n_outputs = n_inputs
#
# learning_rate = 0.01
#
# dropout_rate = 0.3
#
# training = tf.placeholder_with_default(False, shape=(), name='training')
#
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# X_drop = tf.layers.dropout(X, dropout_rate, training=training)
#
# hidden1 = tf.layers.dense(X_drop, n_hidden1, activation=tf.nn.relu)
# hidden2 = tf.layers.dense(hidden1, n_hidden2, activation=tf.nn.relu)
# hidden3 = tf.layers.dense(hidden2, n_hidden3, activation=tf.nn.relu)
# outputs = tf.layers.dense(hidden3, n_outputs)
#
# reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
#
# optimizer = tf.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(reconstruction_loss)
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# n_epochs = 10
# batch_size = 150
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         n_batches = mnist.train.num_examples // batch_size
#         for iteration in range(n_batches):
#             print("\r{}%".format(100 * iteration // n_batches), end=" ")
#             sys.stdout.flush()
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             sess.run(training_op, feed_dict={X: X_batch, training: True})
#         loss_train = reconstruction_loss.eval(feed_dict={X: X_batch})
#         print("\r{}".format(epoch), "Trian MSE:", loss_train)
#         saver.save(sess, "./my_model_stacked_denoising_dropout.ckpt")
#
# show_reconstructed_digits(X, outputs, "./my_model_stacked_denoising_dropout.ckpt")

# Sparse Autoencoder
# reset_graph()
#
# n_inputs = 28 * 28
# n_hidden1 = 1000
# n_outputs = n_inputs
#
#
# def kl_divergence(p, q):
#     return p * tf.log(p / q) + (1 - p) * tf.log((1 - p) / (1 - q))
#
#
# learning_rate = 0.01
# sparsity_target = 0.1
# sparsity_weight = 0.2
#
# X = tf.placeholder(tf.float32, shape=[None, n_inputs])
# #
# # hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.sigmoid)
# # outputs = tf.layers.dense(hidden1, n_outputs)
# # reconstruction_loss = tf.reduce_mean(tf.square(outputs - X))
# #
# hidden1 = tf.layers.dense(X, n_hidden1, activation=tf.nn.sigmoid)
# hidden1_mean = tf.reduce_mean(hidden1, axis=0)
# sparsity_loss = tf.reduce_sum(kl_divergence(sparsity_target, hidden1_mean))
# logits = tf.layers.dense(hidden1, n_outputs)
# outputs = tf.nn.sigmoid(logits)
#
# xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
# reconstruction_loss = tf.reduce_mean(xentropy)
# loss = reconstruction_loss + sparsity_weight * sparsity_loss
#
# optimizer = tf.train.AdamOptimizer(learning_rate)
# training_op = optimizer.minimize(loss)
#
# init = tf.global_variables_initializer()
# saver = tf.train.Saver()
#
# n_epochs = 100
# batch_size = 1000
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         n_batches = mnist.train.num_examples // batch_size
#         for iteration in range(n_batches):
#             print("\r{}%".format(100 * iteration // n_batches), end=" ")
#             sys.stdout.flush()
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             sess.run(training_op, feed_dict={X: X_batch})
#         reconstruction_loss_val, sparsity_loss_val, loss_Val = sess.run([reconstruction_loss, sparsity_loss, loss],
#                                                                         feed_dict={X: X_batch})
#         print("\r{}".format(epoch), 'Train MSE:', reconstruction_loss_val, "\tSparsity losss:", sparsity_loss_val,
#               "\tTotal loss:", loss_Val)
#         saver.save(sess, "./my_model_sparse.ckpt")
#
# show_reconstructed_digits(X, outputs, "./my_model_sparse.ckpt")

# Variational Autoencoder
reset_graph()

n_inputs = 28 * 28
n_hidden1 = 500
n_hidden2 = 500
n_hidden3 = 20
n_hidden4 = n_hidden2
n_hidden5 = n_hidden1
n_outputs = n_inputs
learning_rate = 0.001

initializer = tf.contrib.layers.variance_scaling_initializer()

my_dense_layer = partial(
    tf.layers.dense,
    activation=tf.nn.elu,
    kernel_initializer=initializer
)
X = tf.placeholder(tf.float32, [None, n_inputs])
hidden1 = my_dense_layer(X, n_hidden1)
hidden2 = my_dense_layer(hidden1, n_hidden2)
hidden3_mean = my_dense_layer(hidden2, n_hidden3, activation=None)
# hidden3_sigma = my_dense_layer(hidden2, n_hidden3, activation=None)
# noise = tf.random_normal(tf.shape(hidden3_sigma), dtype=tf.float32)
# hidden3 = hidden3_mean + hidden3_sigma * noise
hidden3_gamma = my_dense_layer(hidden2, n_hidden3, activation=None)
noise = tf.random_normal(tf.shape(hidden3_gamma), dtype=tf.float32)
hidden3 = hidden3_mean + tf.exp(0.5 * hidden3_gamma) * noise
hidden4 = my_dense_layer(hidden3, n_hidden4)
hidden5 = my_dense_layer(hidden4, n_hidden5)
logits = my_dense_layer(hidden5, n_outputs, activation=None)
outputs = tf.sigmoid(logits)

xentropy = tf.nn.sigmoid_cross_entropy_with_logits(labels=X, logits=logits)
reconstuction_loss = tf.reduce_sum(xentropy)

# eps = 1e-10
# latent_loss = 0.5 * tf.reduce_sum(
#     tf.square(hidden3_sigma) + tf.square(hidden3_mean) - 1 - tf.log(eps + tf.square(hidden3_sigma))
# )
latent_loss = 0.5 * tf.reduce_sum(
    tf.exp(hidden3_gamma) + tf.square(hidden3_mean) - 1 - hidden3_gamma
)

loss = reconstuction_loss + latent_loss
optimizer = tf.train.AdamOptimizer(learning_rate)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()
saver = tf.train.Saver()

# n_epochs = 50
# batch_size = 150
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         n_batches = mnist.train.num_examples // batch_size
#         for iteration in range(n_batches):
#             print("\r{}%".format(100 * iteration // n_batches), end=" ")
#             sys.stdout.flush()
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             sess.run(training_op, feed_dict={X: X_batch})
#         loss_val, reconstuction_loss_val, latent_loss_val = sess.run([loss, reconstuction_loss, latent_loss],
#                                                                      feed_dict={X: X_batch})
#         print("\r{}".format(epoch), "Train total loss:", loss_val, "\tReconstruction loss:", reconstuction_loss_val,
#               "\tLatent loss:", latent_loss_val)
#         saver.save(sess, "./my_model_variational.ckpt")

# Generate digits
n_digits = 60
n_epochs = 50
batch_size = 150

# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         n_batches = mnist.train.num_examples // batch_size
#         for iteration in range(n_batches):
#             print("\r{}%".format(100 * iteration // n_batches), end=" ")
#             sys.stdout.flush()
#             X_batch, y_batch = mnist.train.next_batch(batch_size)
#             sess.run(training_op, feed_dict={X: X_batch})
#         loss_val, reconstuction_loss_val, latent_loss_val = sess.run([loss, reconstuction_loss, latent_loss],
#                                                                      feed_dict={X: X_batch})
#         print("\r{}".format(epoch), "Trian total loss:", loss_val, "\tReconstruction loss:", reconstuction_loss_val,
#               "\tLatent loss:", latent_loss_val)
#         saver.save(sess, "./my_model_variational.ckpt")
#     codings_rnd = np.random.normal(size=[n_digits, n_hidden3])
#     outputs_val = outputs.eval(feed_dict={hidden3: codings_rnd})
#
# plt.figure(figsize=(8, 50))
# for iteration in range(n_digits):
#     plt.subplot(n_digits, 10, iteration+1)
#     plot_image(outputs_val[iteration])
#
# n_rows = 6
# n_cols = 10
# plot_multiple_images(outputs_val.reshape(-1, 28, 28), n_rows, n_cols)
# save_fig("generated_digits_plot")
# plt.show()

n_digits = 3
X_test, y_test = mnist.test.next_batch(batch_size)
codings = hidden3

# Encode
with tf.Session() as sess:
    saver.restore(sess, "./my_model_variational.ckpt")
    codings_val = codings.eval(feed_dict={X: X_test})

# Decode
with tf.Session() as sess:
    saver.restore(sess, "./my_model_variational.ckpt")
    outputs_val = outputs.eval(feed_dict={codings: codings_val})

fig = plt.figure(figsize=(8, 2.5 * n_digits))
for iteration in range(n_digits):
    plt.subplot(n_digits, 2, 1 + 2 * iteration)
    plot_image(X_test[iteration])
    plt.subplot(n_digits, 2, 2 + 2 * iteration)
    plot_image(outputs_val[iteration])
plt.show()

# Interpolate digits
n_iterations = 3
n_digits = 6
codings_rnd = np.random.normal(size=[n_digits, n_hidden3])

with tf.Session() as sess:
    saver.restore(sess, "./my_model_variational.ckpt")
    target_codings = np.roll(codings_rnd, -1, axis=0)
    for iteration in range(n_iterations + 1):
        codings_interpolate = codings_rnd + (target_codings - codings_rnd) * iteration / n_iterations
        outputs_val = outputs.eval(feed_dict={codings: codings_interpolate})
        for digit_index in range(n_digits):
            plt.subplot(1, n_digits, digit_index + 1)
            plot_image(outputs_val[digit_index])
        plt.show()
