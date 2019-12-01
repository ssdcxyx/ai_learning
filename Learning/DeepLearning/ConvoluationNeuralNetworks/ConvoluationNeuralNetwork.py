# -*- coding: utf-8 -*-
# @Time    : 2019-01-05 17:28
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : ConvoluationNeuralNetwork.py

from io import open

import numpy as np
import os
import tensorflow as tf
from sklearn.datasets import load_sample_image
import matplotlib.image as mpimg
import sys
import tarfile
from six.moves import urllib
import re
from tensorflow.contrib.slim.nets import inception
import tensorflow.contrib.slim as slim
from collections import defaultdict
from scipy.misc import imresize
from random import sample


def reset_graph(seed=42):
    tf.reset_default_graph()
    tf.set_random_seed(seed)
    np.random.seed(seed)


import matplotlib
import matplotlib.pyplot as plt
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

PROJECT_ROOT_DIT = "."
CHAPTER_ID = "cnn"


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIT, "images", CHAPTER_ID, fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format='png', dpi=300)


def plot_image(image):
    plt.imshow(image, cmap="gray", interpolation="nearest")
    plt.axis("off")


def plot_color_image(image):
    plt.imshow(image.astype(np.uint8), interpolation="nearest")
    plt.axis("off")

# 卷积核
# china = load_sample_image("china.jpg")
# flower = load_sample_image("flower.jpg")
# image = china[150:220, 130:250]
# height, width, channels = image.shape
# image_grayscale = image.mean(axis=2).astype(np.float32)
# images = image_grayscale.reshape(1, height, width, 1)
#
# fmap = np.zeros(shape=(7, 7, 1, 2), dtype=np.float32)
# fmap[:, 3, 0, 0] = 1
# fmap[3, :, 0, 1] = 1
# plot_image(fmap[:, :, 0, 0])
# plt.show()
# plot_image(fmap[:, :, 0, 1])
# plt.show()

# reset_graph()
#
# X = tf.placeholder(tf.float32, shape=(None, height, width, 1))
# feature_maps = tf.constant(fmap)
# convolution = tf.nn.conv2d(X, feature_maps, strides=[1, 1, 1, 1], padding="SAME")
#
# with tf.Session() as sess:
#     output = convolution.eval(feed_dict={X: images})

# plot_image(images[0, :, :, 0])
# save_fig("china_original", tight_layout=False)
# plt.show()

# plot_image(output[0, :, :, 0])
# save_fig("china_vertical", tight_layout=False)
# plt.show()
#
# plot_image(output[0, :, :, 1])
# save_fig("china_horizontal", tight_layout=False)
# plt.show()


# china = load_sample_image("china.jpg")
# flower = load_sample_image("flower.jpg")
# dataset = np.array([china, flower], dtype=np.float32)
# batch_size, height, width, channels = dataset.shape

# filters = np.zeros(shape=(7, 7, channels, 2), dtype=np.float32)
# filters[:, 3, :, 0] = 1
# filters[3, :, :, 1] = 1
#
# X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
# convolution = tf.nn.conv2d(X, filters, strides=[1, 2, 2, 1], padding="SAME")
#
# with tf.Session() as sess:
#     output = sess.run(convolution, feed_dict={X: dataset})
#
# # plt.imshow(output[0, :, :, 1], cmap="gray")
# # plt.show()
#
# for image_index in (0, 1):
#     for feature_map_index in (0, 1):
#         plot_image(output[image_index, :, :, feature_map_index])
#         plt.show()

# reset_graph()
#
# X = tf.placeholder(shape=(None, height, width, channels), dtype=tf.float32)
# conv = tf.layers.conv2d(X, filters=2, kernel_size=7, strides=[2, 2], padding="SAME")
#
# init = tf.global_variables_initializer()
#
# with tf.Session() as sess:
#     init.run()
#     output = sess.run(conv, feed_dict={X: dataset})
#
# plt.imshow(output[0, :, :, 1], cmap="gray")
# plt.show()

# reset_graph()
#
# filter_primes = np.array([2., 3., 5., 7., 11., 13.], dtype=np.float32)
# x = tf.constant(np.arange(1, 13+1, dtype=np.float32).reshape([1, 1, 13, 1]))
# filters = tf.constant(filter_primes.reshape(1, 6, 1, 1))
#
# valid_conv = tf.nn.conv2d(x, filters, strides=[1, 1, 5, 1], padding="VALID")
# same_conv = tf.nn.conv2d(x, filters, strides=[1, 1, 5, 1], padding="SAME")
#
# with tf.Session() as sess:
#     print("VALID:\n", valid_conv.eval())
#     print("SAME:\n", same_conv.eval())
#
# print("VALID:")
# print(np.array([1, 2, 3, 4, 5, 6]).T.dot(filter_primes))
# print(np.array([6, 7, 8, 9, 10, 11]).T.dot(filter_primes))
# print("SAME:")
# print(np.array([0, 1, 2, 3, 4, 5]).T.dot(filter_primes))
# print(np.array([5, 6, 7, 8, 9, 10]).T.dot(filter_primes))
# print(np.array([10, 11, 12, 13, 0, 0]).T.dot(filter_primes))

# 池化层
# X = tf.placeholder(tf.float32, shape=(None, height, width, channels))
# max_pool = tf.nn.max_pool(X, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#
# with tf.Session() as sess:
#     output = sess.run(max_pool, feed_dict={X: dataset})
#
# plt.imshow(output[0].astype(np.uint8))
# plt.show()
#
# plot_color_image(dataset[0])
# save_fig("china_original")
# plt.show()
#
# plot_color_image(output[0])
# save_fig("china_max_pool")
# plt.show()

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, 28 * 28) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, 28 * 28) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_valid, X_train = X_train[:5000], X_train[5000:]
y_valid, y_train = y_train[:5000], y_train[5000:]


def shuffle_batch(X, y, batch_size):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


# height = 28
# width = 28
# channels = 1
# n_inputs = height * width * channels
#
# conv1_fmaps = 32
# conv1_ksize = 3
# conv1_stride = 1
# conv1_pad = "SAME"
#
# conv2_fmaps = 64
# conv2_ksize = 3
# conv2_stride = 2
# conv2_pad = "SAME"
#
# pool3_fmaps = conv2_fmaps
#
# n_fc1 = 64
# n_outputs = 10
#
# reset_graph()
#
# with tf.name_scope("inputs"):
#     X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
#     X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#     y = tf.placeholder(tf.int32, shape=[None], name="y")
#
# conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize,
#                          strides=conv1_stride, padding=conv1_pad,
#                          activation=tf.nn.relu, name="conv1")
# conv2 = tf.layers.conv2d(X_reshaped, filters=conv2_fmaps, kernel_size=conv2_ksize,
#                          strides=conv2_stride, padding=conv2_pad,
#                          activation=tf.nn.relu, name="conv2")
#
# with tf.name_scope("pool3"):
#     pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#     pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 7 * 7])
#
# with tf.name_scope("fc1"):
#     fc1 = tf.layers.dense(pool3_flat, n_fc1, activation=tf.nn.relu, name="fc1")
#
# with tf.name_scope("output"):
#     logits = tf.layers.dense(fc1, n_outputs, name="output")
#     Y_proba = tf.nn.softmax(logits, name="Y_proba")
#
# with tf.name_scope("train"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
#     loss = tf.reduce_mean(xentropy)
#     optimizer = tf.train.AdamOptimizer()
#     training_op = optimizer.minimize(loss)
#
# with tf.name_scope("eval"):
#     correct = tf.nn.in_top_k(logits, y, 1)
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# with tf.name_scope("init_and_save"):
#     init = tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
# n_epochs = 10
# batch_size = 100
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch})
#         acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#         print(epoch, "Last batch accuracy", acc_batch, "Test accuracy", acc_test)
#
#         save_path = saver.save(sess, "./my_mnist_model")

# height = 28
# width = 28
# channels = 1
# n_inputs = height * width
#
# conv1_fmaps = 32
# conv1_ksize = 3
# conv1_stride = 1
# conv1_pad = "SAME"
#
#
# conv2_fmaps = 64
# conv2_ksize = 3
# conv2_stride = 1
# conv2_pad = "SAME"
# conv2_dropout_rate = 0.25
#
# pool3_fmaps = conv2_fmaps
#
# n_fc1 = 29
# fc1_dropout_rate = 0.5
#
# n_outputs = 10
#
# reset_graph()
#
# with tf.name_scope("inputs"):
#     X = tf.placeholder(tf.float32, shape=[None, n_inputs], name="X")
#     X_reshaped = tf.reshape(X, shape=[-1, height, width, channels])
#     y = tf.placeholder(tf.int32, shape=[None], name="y")
#     training = tf.placeholder_with_default(False, shape=[], name="training")
#
# conv1 = tf.layers.conv2d(X_reshaped, filters=conv1_fmaps, kernel_size=conv1_ksize, strides=conv1_stride,
#                          padding=conv1_pad, activation=tf.nn.relu, name="conv1")
# conv2 = tf.layers.conv2d(conv1, filters=conv2_fmaps, kernel_size=conv2_ksize, strides=conv2_stride,
#                          padding=conv2_pad, activation=tf.nn.relu, name="conv2")
#
# with tf.name_scope("pool3"):
#     pool3 = tf.nn.max_pool(conv2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="VALID")
#     pool3_flat = tf.reshape(pool3, shape=[-1, pool3_fmaps * 14 * 14])
#     pool3_flat_drop = tf.layers.dropout(pool3_flat, conv2_dropout_rate, training=training)
#
# with tf.name_scope("fc1"):
#     fc1 = tf.layers.dense(pool3_flat_drop, n_fc1, activation=tf.nn.relu, name="fc1")
#     fc1_drop = tf.layers.dropout(fc1, fc1_dropout_rate, training=training)
#
# with tf.name_scope("output"):
#     logits = tf.layers.dense(fc1, n_outputs, name="output")
#     Y_proba = tf.nn.softmax(logits, name="Y_proba")
#
# with tf.name_scope("train"):
#     xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
#     loss = tf.reduce_mean(xentropy)
#     optimizer = tf.train.AdamOptimizer()
#     training_op = optimizer.minimize(loss)
#
# with tf.name_scope("eval"):
#     correct = tf.nn.in_top_k(logits, y, 1)
#     accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))
#
# with tf.name_scope("init_and_save"):
#     init =tf.global_variables_initializer()
#     saver = tf.train.Saver()
#
#
# def get_model_params():
#     gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
#     return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}
#
#
# def restore_model_params(model_params):
#     gvar_names = list(model_params.keys())
#     assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
#                   for gvar_name in gvar_names}
#     init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
#     feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
#     tf.get_default_session().run(assign_ops, feed_dict=feed_dict)
#
#
# n_epochs = 1000
# batch_size = 50
# iteration = 0
#
# best_loss_val = np.infty
# check_interval = 500
# checks_since_last_progress = 0
# max_checks_without_progress = 20
# best_model_params = None
#
# with tf.Session() as sess:
#     init.run()
#     for epoch in range(n_epochs):
#         for X_batch, y_batch in shuffle_batch(X_train, y_train, batch_size):
#             iteration += 1
#             sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
#             if iteration % check_interval == 0:
#                 loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid})
#                 if loss_val < best_loss_val:
#                     best_loss_val = loss_val
#                     checks_since_last_progress = 0
#                     best_model_params = get_model_params()
#                 else:
#                     checks_since_last_progress += 1
#         acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
#         acc_val = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
#         print("Epoch {}, last batch accuracy: {:.4f}%, valid accuracy: {:.4f}%, valid best loss: {:.6f}".format(
#             epoch, acc_batch * 100, acc_val * 100, best_loss_val
#         ))
#         if checks_since_last_progress > max_checks_without_progress:
#             print("Early stopping!")
#             break
#     if best_model_params:
#         restore_model_params(best_model_params)
#     acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
#     print("Final accuracy on test set:", acc_test)
#     save_path = saver.save(sess, "./my_mnist.model")


width = 299
height = 299
channels = 3

# test_image = mpimg.imread(os.path.join("images", "cnn", "test_image.png"))[:, :, :channels]
# plt.imshow(test_image)
# plt.axis("off")
# plt.show()
#
# test_image = 2 * test_image - 1
#
TF_MODELS_URL = "http://download.tensorflow.org/models"
INCEPTION_V3_URL = TF_MODELS_URL + "/inception_v3_2016_08_28.tar.gz"
INCEPTION_PATH = os.path.join("garbage", "inception")
INCEPTION_V3_CHECKPOINT_PATH = os.path.join(INCEPTION_PATH, "inception_v3.ckpt")


def download_progress(count, block_size, total_size):
    percent = count * block_size * 100 // total_size
    sys.stdout.write("\rDownloading:{}%".format(percent))
    sys.stdout.flush()


# def fetch_pertrained_inception_v3(url=INCEPTION_V3_URL, path=INCEPTION_PATH):
#     if os.path.exists(INCEPTION_V3_CHECKPOINT_PATH):
#         return
#     os.makedirs(path, exist_ok=True)
#     tgz_path = os.path.join(path, "inception_v3.tgz")
#     urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
#     inception_tgz = tarfile.open(tgz_path)
#     inception_tgz.extractall(path=path)
#     inception_tgz.close()
#     os.remove(tgz_path)
#
#
# # fetch_pertrained_inception_v3()
#
# CLASS_NAME_REGEX = re.compile(r"^n\d+\s+(.*)\s*$", re.M | re.U)
#
#
# def load_class_names():
#     path = os.path.join("garbage", "inception", "imagenet_class_names.txt")
#     with open(path, encoding="utf-8") as f:
#         content = f.read()
#         return CLASS_NAME_REGEX.findall(content)
#
#
# class_names = ["background"] + load_class_names()
# print(class_names[:5])
#
# reset_graph()
#
# X = tf.placeholder(tf.float32, shape=[None, 299, 299, 3], name="X")
# with slim.arg_scope(inception.inception_v3_arg_scope()):
#     logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=False)
# predictions = end_points["Predictions"]
# saver = tf.train.Saver()
#
# with tf.Session() as sess:
#     saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
#
# X_test = test_image.reshape(-1, height, width, channels)
#
# with tf.Session() as sess:
#     saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
#     prediction_val = predictions.eval(feed_dict={X: X_test})
#
# most_likely_class_index = np.argmax(prediction_val[0])
# print(most_likely_class_index)
# print(class_names[most_likely_class_index])
#
# top_5 = np.argpartition(prediction_val[0], -5)[-5:]
# top_5 = reversed(top_5[np.argsort(prediction_val[0][top_5])])
# for i in top_5:
#     print("{0}:{1:.2f}%".format(class_names[i], 100 * prediction_val[0][i]))


FLOWERS_URL = "http://download.tensorflow.org/example_images/flower_photos.tgz"
FLOWERS_PATH = os.path.join("garbage", "flowers")


def fetch_flowers(url=FLOWERS_URL, path=FLOWERS_PATH):
    if os.path.exists(FLOWERS_PATH):
        return
    os.makedirs(path, exist_ok=True)
    tgz_path = os.path.join(path, "flower_photo.tgz")
    urllib.request.urlretrieve(url, tgz_path, reporthook=download_progress)
    flower_tgz = tarfile.open(tgz_path)
    flower_tgz.extractall(path=path)
    flower_tgz.close()
    os.remove(tgz_path)


# fetch_flowers()

flower_root_path = os.path.join(FLOWERS_PATH, "flower_photos")
flower_classes = sorted([dirname for dirname in os.listdir(flower_root_path)
                         if os.path.isdir(os.path.join(flower_root_path, dirname))])
# print(flower_classes)

image_paths = defaultdict(list)

for flower_class in flower_classes:
    image_dir = os.path.join(flower_root_path, flower_class)
    for filepath in os.listdir(image_dir):
        if filepath.endswith(".jpg"):
            image_paths[flower_class].append(os.path.join(image_dir, filepath))

for paths in image_paths.values():
    paths.sort()

n_examples_per_class = 2

for flower_class in flower_classes:
    # print("Class:", flower_class)
    # plt.figure(figsize=(10, 5))
    for index, example_image_path in enumerate(image_paths[flower_class][:n_examples_per_class]):
        example_image = mpimg.imread(example_image_path)[:, :, :channels]
        # plt.subplot(100 + n_examples_per_class * 10 + index + 1)
        # plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
        # plt.imshow(example_image)
        # plt.axis("off")
    # plt.show()


def prepare_image(image, target_width=299, target_height=299, max_zoom=0.2):
    height = image.shape[0]
    width = image.shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = width if crop_vertically else int(height * target_image_ratio)
    crop_height = int(width / target_image_ratio) if crop_vertically else height
    resize_factor = np.random.rand() * max_zoom + 1.0
    crop_width = int(crop_width / resize_factor)
    crop_height = int(crop_height / resize_factor)

    x0 = np.random.randint(0, width - crop_width)
    y0 = np.random.randint(0, height - crop_height)
    x1 = x0 + crop_width
    y1 = y0 + crop_height

    image = image[y0:y1, x0: x1]

    if np.random.rand() < 0.5:
        image = np.fliplr(image)
    image = imresize(image, (target_width, target_height))
    return image.astype(np.float32) / 255


# plt.figure(figsize=(6, 8))
# plt.imshow(example_image)
# plt.title("{}x{}".format(example_image.shape[1], example_image.shape[0]))
# plt.axis("off")
# plt.show()
#
# prepared_image = prepare_image(example_image)
# plt.figure(figsize=(8, 8))
# plt.imshow(prepared_image)
# plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
# plt.axis("off")
# plt.show()
#
# rows, cols = 2, 3
#
# plt.figure(figsize=(14, 8))
# for row in range(rows):
#     for col in range(cols):
#         prepared_image = prepare_image(example_image)
#         plt.subplot(rows, cols, row * cols + col + 1)
#         plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
#         plt.imshow(prepared_image)
#         plt.axis("off")
# plt.show()


def prepare_image_with_tensorflow(image, target_width=299, target_height=299, max_zoom=0.2):
    image_shape = tf.cast(tf.shape(image), tf.float32)
    height = image_shape[0]
    width = image_shape[1]
    image_ratio = width / height
    target_image_ratio = target_width / target_height
    crop_vertically = image_ratio < target_image_ratio
    crop_width = tf.cond(crop_vertically,
                         lambda: width,
                         lambda: height * target_image_ratio)
    crop_height = tf.cond(crop_vertically,
                          lambda: width / target_image_ratio,
                          lambda: height)
    resize_factor = tf.random_uniform(shape=[], minval=1.0, maxval=1.0 + max_zoom)
    crop_width = tf.cast(crop_width / resize_factor, tf.int32)
    crop_height = tf.cast(crop_height / resize_factor, tf.int32)
    box_size = tf.stack([crop_height, crop_width, 3])
    image = tf.random_crop(image, box_size)
    image = tf.image.random_flip_left_right(image)
    image_batch = tf.expand_dims(image, 0)
    image_batch = tf.image.resize_bilinear(image_batch, [target_height, target_width])
    image = image_batch[0] / 255
    return image


reset_graph()

input_image = tf.placeholder(tf.uint8, shape=[None, None, 3])
prepared_image_op = prepare_image_with_tensorflow(input_image)

# with tf.Session() as sess:
#     prepared_image = prepared_image_op.eval(feed_dict={input_image: example_image})
#
# plt.figure(figsize=(6, 6))
# plt.imshow(prepared_image)
# plt.title("{}x{}".format(prepared_image.shape[1], prepared_image.shape[0]))
# plt.axis("off")
# plt.show()

reset_graph()

X = tf.placeholder(tf.float32, shape=[None, height, width, channels], name="X")
training = tf.placeholder_with_default(False, shape=[])
with slim.arg_scope(inception.inception_v3_arg_scope()):
    logits, end_points = inception.inception_v3(X, num_classes=1001, is_training=training)
inception_saver = tf.train.Saver()

# print(logits.op.inputs[0])
# print(logits.op.inputs[0].op.inputs[0])
# print(logits.op.inputs[0].op.inputs[0].op.inputs[0])
#
# print(end_points)
# print(end_points["PreLogits"])

prelogits = tf.squeeze(end_points["PreLogits"], axis=[1, 2])

n_outputs = len(flower_classes)

with tf.name_scope("new_output_layer"):
    flower_logits = tf.layers.dense(prelogits, n_outputs, name="flower_logits")
    Y_proba = tf.nn.softmax(flower_logits, name="Y_proba")

y = tf.placeholder(tf.int32, shape=[None])

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=flower_logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    flower_vars = tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope="flower_logits")
    training_op = optimizer.minimize(loss, var_list=flower_vars)

with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(flower_logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

# print([v.name for v in flower_vars])

flower_class_ids = {flower_class: index for index, flower_class in enumerate(flower_classes)}
print(flower_class_ids)

flower_paths_and_classes = []
for flower_class, paths in image_paths.items():
    for path in paths:
        flower_paths_and_classes.append((path, flower_class_ids[flower_class]))

test_ratio = 0.2
train_size = int(len(flower_paths_and_classes) * (1 - test_ratio))
np.random.shuffle(flower_paths_and_classes)

flower_paths_and_classes_train = flower_paths_and_classes[:train_size]
flower_paths_and_classes_test = flower_paths_and_classes[train_size:]

# print(flower_paths_and_classes[:3])


def prepare_batch(flower_paths_and_classes, batch_size):
    batch_paths_and_classes = sample(flower_paths_and_classes, batch_size)
    images = [mpimg.imread(path)[:, :, :channels] for path, labels in batch_paths_and_classes]
    prepared_images = [prepare_image(image) for image in images]
    X_batch = 2 * np.stack(prepared_images) - 1
    y_batch = np.array([labels for path, labels in batch_paths_and_classes], dtype=np.int32)
    return X_batch, y_batch


X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size=4)
# print(X_batch.shape)
# print(X_batch.dtype)
# print(y_batch.shape)
# print(y_batch.dtype)

X_test, y_test = prepare_batch(flower_paths_and_classes_test, batch_size=len(flower_paths_and_classes_test))
# print(X_test.shape)

n_epochs = 10
batch_size = 40
n_iteration_per_epochs = len(flower_paths_and_classes_train) // batch_size

with tf.Session() as sess:
    init.run()
    inception_saver.restore(sess, INCEPTION_V3_CHECKPOINT_PATH)
    for epoch in range(n_epochs):
        print("Epoch", epoch, end="")
        for iteration in range(n_iteration_per_epochs):
            print(".", end="")
            X_batch, y_batch = prepare_batch(flower_paths_and_classes_train, batch_size)
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        print("Last batch accuracy:", acc_batch)
        save_path = saver.save(sess, "./my_flowers_model")

n_test_batches = 10
X_test_batches = np.array_split(X_test, n_test_batches)
y_test_batches = np.array_split(y_test, n_test_batches)

with tf.Session() as sess:
    saver.restore(sess, "./my_flowers_model")
    print("Computing final accuracy on the test set (this will take a while)...")
    acc_test = np.mean([
        accuracy.eval(feed_dict={X: X_test_batch, y: y_test_batch})
        for X_test_batch, y_test_batch in zip(X_test_batches, y_test_batches)
    ])
    print("Test accuracy:",  acc_test)


