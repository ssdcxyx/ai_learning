# -*- coding: utf-8 -*-
# @time       : 11/11/2019 2:20 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : LeNet.py
# @description: 

import tensorflow as tf
import numpy as np

HEIGHT, WIDTH, CHANNELS = 28, 28, 1
NUM_CLASSES = 10

(X_train, y_train), (X_test, y_test) = tf.keras.datasets.mnist.load_data()
X_train = X_train.astype(np.float32).reshape(-1, HEIGHT * WIDTH) / 255.0
X_test = X_test.astype(np.float32).reshape(-1, HEIGHT * WIDTH) / 255.0
y_train = y_train.astype(np.int32)
y_test = y_test.astype(np.int32)
X_train, y_train = X_train[5000:], y_train[5000:]
X_valid, y_valid = X_train[:5000], y_train[:5000]

with tf.name_scope('inputs'):
    X = tf.placeholder(dtype=tf.float32, shape=[None, HEIGHT * WIDTH * CHANNELS], name="X")
    X_reshape = tf.reshape(X, shape=[-1, HEIGHT, WIDTH, CHANNELS])
    y = tf.placeholder(dtype=tf.int32, shape=[None], name="y")
    training = tf.placeholder_with_default(False, shape=[], name="training")


with tf.name_scope('dnn'):
    conv1 = tf.layers.conv2d(inputs=X_reshape,
                             kernel_size=[5, 5], filters=6, strides=[1, 1],
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             use_bias=True, bias_initializer=tf.constant_initializer(0.0),
                             activation='relu', name="conv1")
    pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                    pool_size=[2, 2], strides=[2, 2],
                                    name="pool1")
    conv2 = tf.layers.conv2d(inputs=pool1,
                             kernel_size=[5, 5], filters=16, strides=[1, 1],
                             kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             use_bias=True, bias_initializer=tf.constant_initializer(0.0),
                             activation='relu', name="conv2")
    pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                    pool_size=[2, 2], strides=[2, 2],
                                    name="pool2")
    flatten1 = tf.layers.flatten(inputs=pool2, name="flatten1")
    dropout1 = tf.layers.dropout(inputs=flatten1, rate=0.25, training=training, name="dropout1")

    dense1 = tf.layers.dense(inputs=dropout1,
                             units=120, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             use_bias=True, bias_initializer=tf.constant_initializer(0.0),
                             activation='relu', name='dense1')

    dropout2 = tf.layers.dropout(inputs=dense1, rate=0.25, training=training, name="dropout2")
    dense2 = tf.layers.dense(inputs=dropout2,
                             units=84, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             use_bias=True, bias_initializer=tf.constant_initializer(0.0),
                             activation='relu', name='dense2')
    dropout3 = tf.layers.dropout(inputs=dense2, rate=0.25, training=training, name="dropout2")
    logits = tf.layers.dense(inputs=dropout3,
                             units=NUM_CLASSES, kernel_initializer=tf.truncated_normal_initializer(stddev=0.1),
                             use_bias=True, bias_initializer=tf.constant_initializer(0.0),
                             name='output')
    Y_proba = tf.nn.softmax(logits, name="Y_paoba")

with tf.name_scope('train'):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimier = tf.train.AdamOptimizer(learning_rate=1e-3)
    training_op = optimier.minimize(loss)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


n_epochs = 100
batch_size = 32
iteration = 0

best_loss_val = np.infty
# 检查间隔
check_interval = 500
check_since_last_process = 0
# 损失未下降的最大轮数
max_checks_without_progress = 20
best_model_params = None


def shuffle_batch(X, y):
    rnd_idx = np.random.permutation(len(X))
    n_batches = len(X) // batch_size
    for batch_idx in np.array_split(rnd_idx, n_batches):
        X_batch, y_batch = X[batch_idx], y[batch_idx]
        yield X_batch, y_batch


def get_model_params():
    gvars = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES)
    return {gvar.op.name: value for gvar, value in zip(gvars, tf.get_default_session().run(gvars))}


def restore_model_params(model_params):
    gvar_names = list(model_params.keys())
    assign_ops = {gvar_name: tf.get_default_graph().get_operation_by_name(gvar_name + "/Assign")
                  for gvar_name in gvar_names}
    init_values = {gvar_name: assign_op.inputs[1] for gvar_name, assign_op in assign_ops.items()}
    feed_dict = {init_values[gvar_name]: model_params[gvar_name] for gvar_name in gvar_names}
    tf.get_default_session().run(assign_ops, feed_dict=feed_dict)


with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        for X_batch, y_batch in shuffle_batch(X_train, y_train):
            iteration += 1
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
            if iteration % check_interval == 0:
                loss_val = loss.eval(feed_dict={X: X_valid, y: y_valid})
                if loss_val < best_loss_val:
                    best_loss_val = loss_val
                    check_since_last_process = 0
                    best_model_params = get_model_params()
                else:
                    check_since_last_process += 1
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_valid = accuracy.eval(feed_dict={X: X_valid, y: y_valid})
        print("Epoch {}, last batch accuracy: {:.4f}%, valid accuracy:{:.4f}% valid best loss :{:.6f}".format(
            epoch, acc_batch * 100, acc_valid * 100, best_loss_val
        ))
        if check_since_last_process > max_checks_without_progress:
            print("Eary stopping!")
            break
    if best_model_params:
        restore_model_params(best_model_params)
    acc_test = accuracy.eval(feed_dict={X: X_test, y: y_test})
    print("Final accuracy on test set: ", acc_test)
    save_path = saver.save(sess, "./my_mnist")








