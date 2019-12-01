# -*- coding: utf-8 -*-
# @Time    : 2019-06-30 08:52
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Test.py

import sklearn

if __name__ == '__main__':
    mean = (7921 + 5184 + 8836 + 4761) / 4
    max_min = 8836 - 4761
    print((4761 - mean)/max_min)
