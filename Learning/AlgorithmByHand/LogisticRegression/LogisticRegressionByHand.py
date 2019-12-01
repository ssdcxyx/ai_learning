# -*- coding: utf-8 -*-
# @time       : 2019-10-20 13:51
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : LogisticRegressionByHand.py
# @description: 

from random import sample, normalvariate, shuffle
from math import exp

from sklearn import linear_model

from Learning.AlgorithmByHand.Evaulte import run_time, get_accuracy, get_precision, get_recall, get_auc
from Learning.AlgorithmByHand.Preprocess import load_data, min_max_scale, train_test_split
from Learning.AlgorithmByHand import DataPath


def model_evaluation(clf, X, y, mode="hand"):
    if mode == "sklearn":
        y_hat = clf.predict(X)
        y_hat_prob = clf.predict_proba(X)[:, 1]
    else:
        y_hat = clf.predict(X)
        y_hat_prob = [clf._predict(Xi) for Xi in X]
    ret = dict()
    ret["Accuracy"] = get_accuracy(y, y_hat)
    ret["Precision"] = get_precision(y, y_hat)
    ret["Recall"] = get_recall(y, y_hat)
    ret['AUC'] = get_auc(y, y_hat_prob)
    for k, v in ret.items():
        print("%s:%.3f" % (k, v))


def sigmoid(x, x_min=-100):
    return 1 / (1 + exp(-x)) if x > x_min else 0


class RegressionBase(object):
    def __init__(self):
        self.weights = None
        self.bias = None

    def _predict(self, Xi):
        return NotImplemented

    def get_gradient_descent(self, Xi, yi):
        yi_hat = self._predict(Xi)
        weights_grad_delta = [(yi - yi_hat) * Xij for Xij in Xi]
        bias_grad_delta = yi - yi_hat
        return weights_grad_delta, bias_grad_delta

    def shuffle_batch(self, X, y, batch_size):
        shuffle(X)
        iters = len(X) // batch_size
        for iter in range(iters):
            start = iter * batch_size
            end = (iter + 1) * batch_size
            yield X[start: end], y[start: end]

    # 随机梯度下降
    def stochastic_gradient_descent(self, X, y, lr, epochs, batch_size):
        m, n = len(X), len(X[0])
        # 初始化权重
        self.weights = [normalvariate(0, 0.01) for _ in range(n)]
        self.bias = 0
        for _ in range(epochs):
            for X_batch, y_batch in self.shuffle_batch(X, y, batch_size):
                weights_grad = [0 for _ in range(n)]
                bias_grad = 0
                for i in range(len(X_batch)):
                    # 计算梯度
                    weights_grad_delta, bias_grad_delta = self.get_gradient_descent(X_batch[i], y_batch[i])
                    weights_grad = [w_d + w_g_d for w_d, w_g_d in zip(weights_grad, weights_grad_delta)]
                    bias_grad += bias_grad_delta
                    # 更新权重
                self.weights = [w + lr * w_g * 2 / batch_size for w, w_g in zip(self.weights, weights_grad)]
                self.bias += lr * bias_grad * 2 / batch_size

    def fit(self, X, y, lr, epochs, batch_size=32):
        self.stochastic_gradient_descent(X, y, lr, epochs, batch_size)

    def predict(self, X):
        return NotImplemented


class LogisticRegression(RegressionBase):
    def __init__(self):
        RegressionBase.__init__(self)

    def _predict(self, Xi):
        z = sum(wi * xij for wi, xij in zip(self.weights, Xi)) + self.bias
        return sigmoid(z)

    def predict(self, X, threshold=0.5):
        return [int(self._predict(Xi) >= threshold) for Xi in X]


def main():
    X, y = load_data(DataPath.BREAST_CANCER)
    X = min_max_scale(X)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=10)

    @run_time
    def stochastic():
        print("Testing the performance of LogisticRegression(stochastic)...")
        clf = LogisticRegression()
        clf.fit(X=X_train, y=y_train, lr=0.01, epochs=200, batch_size=32)
        model_evaluation(clf, X_test, y_test)

    @run_time
    def _sklearn():
        print("Testing the performance of LogisticRegression(sklearn)...")
        clf = linear_model.LogisticRegression(solver='liblinear', max_iter=200, random_state=42)
        clf.fit(X_train, y_train)
        model_evaluation(clf, X_test, y_test, mode="sklearn")

main()



