# -*- coding: utf-8 -*-
# @Time    : 2018-12-22 18:55
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Test.py


def f(x, y):
    return x**2*y + y +2


def derivative(f, x, y, x_eps, y_eps):
    return (f(x + x_eps, y+ y_eps) - f(x, y)) / (x_eps + y_eps)


print(derivative(f, 3, 4, 0.00001, 0))
print(derivative(f, 3, 4, 0, 0.00001))

