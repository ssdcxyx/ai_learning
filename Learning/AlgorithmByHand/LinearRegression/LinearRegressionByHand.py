# -*- coding: utf-8 -*-
# @time       : 2019-10-15 20:30
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : LinearRegressionByHand.py
# @description: 线性回归

from random import sample, normalvariate, shuffle

from sklearn import linear_model


from Learning.AlgorithmByHand.Evaulte import run_time, get_r2
from Learning.AlgorithmByHand.Preprocess import load_data, min_max_scale, train_test_split
from Learning.AlgorithmByHand import DataPath


class LinearRegressionBase(object):
    def __init__(self):
        self.bias = None
        self.weights = None

    def _predict(self, Xi):
        return NotImplemented

    def get_gradient_data(self, Xi, yi):
        yi_hat = self._predict(Xi)
        weights_grad_deta = [(yi - yi_hat) * Xij for Xij in Xi]
        bias_grad_delta = yi - yi_hat
        return weights_grad_deta, bias_grad_delta

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
        self.weights = [normalvariate(0, 1) for _ in range(n)]
        self.bias = 0
        for _ in range(epochs):
            for X_batch, y_batch in self.shuffle_batch(X, y, batch_size):
                weights_grad = [0 for _ in range(n)]
                bias_grad = 0
                for i in range(len(X_batch)):
                    # 计算梯度
                    weights_grad_deta, bias_grad_deta = self.get_gradient_data(X_batch[i], y_batch[i])
                    weights_grad = [w_grad + w_grad_d for w_grad, w_grad_d in zip(weights_grad, weights_grad_deta)]
                    bias_grad += bias_grad_deta
                # 更新权重
                self.bias = self.bias + lr * bias_grad * 2 / batch_size
                self.weights = [w + lr * w_grad * 2 / batch_size for w, w_grad in zip(self.weights, weights_grad)]

    def fit(self, X, y, lr, epochs, batch_size=32):
        self.stochastic_gradient_descent(X, y, lr, epochs, batch_size)

    def predict(self, X):
        return NotImplemented


class LinearRegression(LinearRegressionBase):
    def __init__(self):
        LinearRegressionBase.__init__(self)

    def _predict(self, Xi):
        return sum(wi * xij for wi, xij in zip(self.weights, Xi)) + self.bias

    def predict(self, X):
        return [self._predict(Xi) for Xi in X]


def main():
    X, y = load_data(DataPath.HOUSING_PRICE)
    X = min_max_scale(X)
    X_train, y_train, X_test, y_test = train_test_split(X, y, test_size=0.3, random_state=10)
    @run_time
    def stochastic():
        print("Testing the performance of LinearRegression(stochastic)...")
        reg = LinearRegression()
        reg.fit(X=X_train, y=y_train, lr=0.001, epochs=5000, batch_size=32)
        print("r2_score:", get_r2(y_test, reg.predict(X_test)))

    @run_time
    def _sklearn():
        print("Testing the performance of LinearRegression(sklearn)...")
        reg = linear_model.LinearRegression()
        reg.fit(X_train, y_train)
        print("r2_score:", get_r2(y_test, reg.predict(X_test)))


main()
