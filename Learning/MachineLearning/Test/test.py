import tensorflow as tf
import csv
import numpy as np


def loadCSV(filename):
    # 加载数据，一行行的存入列表
    dataSet = []
    with open(filename, 'r', encoding='gbk') as file:
        csvReader = csv.reader(file)
        for line in csvReader:
            if csvReader.line_num==1:
                continue
            dataSet.append(line)

    # 列都转换为float类型
    featLen = len(dataSet[0])
    for data in dataSet:
        for column in range(featLen):
            data[column] = float(data[column].strip())
    return dataSet


def addLayer(inputData, inSize, outSize, activity_function=None):
    Weights = tf.Variable(tf.zeros([inSize, outSize]))
    basis = tf.Variable(tf.zeros([1, outSize]) + 0.1)
    weights_plus_b = tf.matmul(inputData, Weights) + basis
    if activity_function is None:
        ans = weights_plus_b
    else:
        ans = activity_function(weights_plus_b)
    return ans


def neuron_layer(X, n_neurons, name, activation=None):
    with tf.name_scope(name):
        n_inputs = int(X.get_shape()[1])
        stddev = 2 / np.sqrt(n_inputs)
        init = tf.truncated_normal((n_inputs, n_neurons), stddev=stddev)
        W = tf.Variable(init, name="weights")
        b = tf.Variable(tf.zeros([n_neurons]), name="biases")
        z = tf.matmul(X, W) + b
        if activation == "relu":
            return tf.nn.relu(z)
        else:
            return z


x_data = loadCSV('img/training.csv')
y_data = loadCSV('img/testing.csv')

X = tf.placeholder(tf.float32, shape=(None, 3), name="X")
y = tf.placeholder(tf.float32, shape=(None), name="y")

hidden1 = tf.layers.dense(X, 1, activation=tf.nn.relu, name="hidden1")
logits = tf.layers.dense(hidden1, 1, name="outputs")
loss = tf.reduce_mean(tf.square(y-logits), name="loss")

train = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

with tf.Session() as sess:
    init = tf.initialize_all_variables()
    sess.run(init)
    for i in range(150):
        sess.run(train, feed_dict={X: x_data, y: y_data})
        if i % 10 == 0:
            print(sess.run(loss, feed_dict={X: x_data, y: y_data}))
