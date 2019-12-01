# -*- coding: utf-8 -*-
# @Time    : 2019-05-10 10:27
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : DatVisual.py

import matplotlib
import matplotlib.pyplot as plt
import math
# 绘图设置
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

import os
import numpy as np
import pandas as pd

from setting import Constant
from handle import GetData
from handle import DataProcess
from handle import FeatureExtraction
from setting.tool import sample


def save_fig(fig_id, tight_layout=True):
    path = os.path.join(Constant.PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)


def show_data_distribution(data, features, sample_size=100):
    # 随机抽样
    sample = np.random.permutation(len(data))
    data = data.take(sample[:sample_size])
    for index, row in data[features].iteritems():
        fig = plt.figure(index, figsize=(20, 14))
        ax = plt.subplot(1, 1, 1)
        plt.scatter(row.index, row.values, cmap="coolwarm")
        plt.xlabel("label")
        plt.ylabel("value")
        ax.set_title(index)
        plt.show()


# 显示用户增长趋势
def show_user_growth_trend(datas, columns):
    fig = plt.figure('user_growth_trend', figsize=(20, 14))
    ax = plt.subplot(1, 1, 1)
    for i in range(0, len(datas)):
        # X轴数据
        xs = []
        sum_user = 0
        for index, row in (datas[i])[columns[i]].iteritems():
            sum_user += row
            xs.append(sum_user)
        plt.plot(pd.to_datetime(datas[i].report_date), xs,  "-", label=columns[i])
    plt.xlabel("date")
    plt.ylabel("amount")
    ax.set_title('user_growth_trend')
    plt.legend()
    plt.show()


# 显示TSNE降维之后用户分类情况
def show_classification_distribution(data, target):
    plt.figure(figsize=(12, 6))
    plt.subplot(111)
    plt.scatter(data[:, 0], data[:, 1], c=target)
    plt.colorbar()
    plt.show()


# 抽样显示各类用户的数据
def show_kinds_of_user_data(data):
    kind1_user = data[data["label"] == 0]
    kind2_user = data[data["label"] == 1]
    kind3_user = data[data["label"] == 2]
    kind4_user = data[data["label"] == 3]
    print(sample(kind1_user))
    print(sample(kind2_user))
    print(sample(kind3_user))
    print(sample(kind4_user))


# 绘制星座映射图
def show_constellation_mapping():
    theta = np.arange(0, 2*np.pi, 0.01)
    x = 0 + 1 * np.cos(theta)
    y = 0 + 1 * np.sin(theta)
    # 等分点
    xs = [0, 1/2, math.sqrt(3)/2, 1, math.sqrt(3)/2, 1/2, 0, -1/2, -math.sqrt(3)/2, -1, -math.sqrt(3)/2, -1/2]
    ys = [1, math.sqrt(3)/2, 1/2, 0, -1/2, -math.sqrt(3)/2, -1, -math.sqrt(3)/2, -1/2, 0, 1/2, math.sqrt(3)/2]
    xs1 = [0, 1/2]
    ys1 = [1, math.sqrt(3)/2]
    xs2= [0, math.sqrt(3)/2]
    ys2 = [1, 1/2]
    xs3 = [0, 0]
    ys3 = [1, -1]
    xs4 = [0, -1/2]
    ys4 = [1, math.sqrt(3)/2]
    fig = plt.figure()
    axes = fig.add_subplot(111)
    axes.plot(x, y)
    axes.scatter(xs, ys)
    axes.plot(xs1, ys1)
    axes.plot(xs2, ys2)
    axes.plot(xs3, ys3)
    axes.plot(xs4, ys4)
    axes.axis("equal")
    plt.show()


if __name__ == "__main__":
    # 用户信息分布(随机采样)
    # user_profile = GetData.get_data(Constant.user_profile_table, encoding="GBK")
    # show_data_distribution(DataProcess.label_quantization(user_profile, columns=['constellation']),
    #                        features=['sex', 'city', 'constellation'])
    # 用户注册与第一次交易增长情况
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y%m%d')
    user_register_growth_data = GetData.get_data(Constant.user_register_growth_table, parse_dates=['report_date'],
                                                 date_parser=dateparse)
    user_first_trading_growth_data = GetData.get_data(Constant.user_first_trading_growth_table,
                                                      parse_dates=['report_date'],
                                                      date_parser=dateparse)

    show_user_growth_trend([user_register_growth_data, user_first_trading_growth_data],
                           columns=['count_user_register_growth', 'count_user_first_trading_growth'])
    show_constellation_mapping()
    print()
