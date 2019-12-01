# -*- coding: utf-8 -*-
# @Time    : 2019-06-30 08:10
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : DataSnooping.py

import pandas as pd
import numpy as np
from pandas import DataFrame,Series
import matplotlib.pyplot as plt

from queue import Queue

from DataPreprocessing import Data
from DataPreprocessing import Feature
from Setting import FilePath
from Other import Tool


# 初探数据
def preliminary_data(data):
    print(data)
    print('_' * 100)
    print(data.info())
    print('_'*100)
    print(data.describe())


# 特征分布
def feature_distribution(data):
    fig = plt.figure()
    fig.set(alpha=0.2)

    plt.subplot2grid((2, 3), (0, 0))
    data[Feature.survived].value_counts().plot(kind='bar')
    plt.title(u"获救情况")
    plt.ylabel(u"人数")

    plt.subplot2grid((2, 3), (0, 1))
    data.Pclass.value_counts().plot(kind='bar')
    plt.title(u"乘客等级分布")
    plt.ylabel(u"人数")

    plt.subplot2grid((2, 3), (0, 2))
    plt.scatter(data[Feature.survived], data.Age)
    plt.ylabel(u"年龄")
    plt.grid(b=True, which='major', axis='y')
    plt.title(u"获救者年龄分布")

    plt.subplot2grid((2, 3), (1, 0), colspan=2)
    data.Age[data.Pclass == 1].plot(kind='kde')
    data.Age[data.Pclass == 2].plot(kind='kde')
    data.Age[data.Pclass == 3].plot(kind='kde')
    plt.xlabel(u"年龄")
    plt.ylabel(u"密度")
    plt.title(u"各等级的乘客年龄分布")
    plt.legend((u'头等舱', u"二等舱", u"三等舱"), loc='best')

    plt.subplot2grid((2, 3), (1, 2))
    data.Embarked.value_counts().plot(kind='bar')
    plt.title(u"各登船口岸上船人数")
    plt.ylabel(u"人数")


# 特征与获救的关联统计
def feature_result_correlation(data, feature, title="", xlabel="", ylabel="", xresult=False):
    surviveds = pd.DataFrame()
    if xresult:
        eigenvalues = list(data.drop_duplicates(subset=feature, keep='first')[feature])
        if feature == Feature.cabin:
            no_cabin = data[Feature.survived][pd.isnull(data[feature])].value_counts()
            cabin = data[Feature.survived][pd.notnull(data[feature])].value_counts()
            surviveds[u'有'] = cabin
            surviveds[u'无'] = no_cabin
        else:
            for eigenvalue in eigenvalues:
                survived = data[Feature.survived][data[feature] == eigenvalue].value_counts()
                surviveds[eigenvalue] = survived
    else:
        no_survived = data[feature][data[Feature.survived] == 0].value_counts()
        survived = data[feature][data[Feature.survived] == 1].value_counts()
        surviveds = pd.DataFrame({u'获救': survived, u'未获救': no_survived})
    surviveds.plot(kind='bar', stacked=True)
    if xresult:
        plt.xlabel('是否获救')
    else:
        plt.xlabel(xlabel)
    plt.title(title)
    plt.ylabel(ylabel)
    plt.grid(axis="y")


# 多特征与获救的关联统计
def multiple_feature_result_correlation(data, features, title):
    fig = plt.figure()
    fig.set(alpha=0.65)
    eigenvalues = []
    for feature in features:
        eigenvalue = list(data.drop_duplicates(subset=feature, keep='first')[feature])
        eigenvalues.append(eigenvalue)
    combinations = Tool.feature_combination(eigenvalues, 0, [])
    for i in range(0, len(combinations)):
        combination_data = data.copy()
        legend = ""
        for j in range(0, len(features)):
            combination_data = combination_data[data[features[j]] == combinations[i][j]]
            legend += features[j] + "=" + str(combinations[i][j]) + "\n"
        pos = len(eigenvalues[0])*100+(len(combinations)/len(eigenvalues[0]))*10 + i + 1
        print(pos)
        if i == 0:
            ax = fig.add_subplot(pos)
        else:
            ax = fig.add_subplot(pos, sharey=ax)
        combination_data[Feature.survived].value_counts().plot(kind='bar')
        ax.set_xticklabels([u"未获救", u"获救"], rotation=0)
        plt.legend([legend], loc='best')
        plt.title(title)
        plt.xlabel(u"是否获救")
        plt.ylabel(u"人数")
        plt.grid(axis="y")


if __name__ == '__main__':
    train_data = Data.get_data(FilePath.original_train_data)
    # feature_result_correlation(train_data)
    # feature_result_correlation(train_data, feature=Feature.p_class,
    #                     title=u'客舱等级与获救的关联统计', xlabel=u"客舱等级", ylabel=u"人数")
    # feature_result_correlation(train_data, feature=Feature.sex,
    #                     title=u'性别与获救的关联统计', xlabel=u'性别', ylabel=u'人数')
    # feature_result_correlation(train_data, feature=Feature.embarked,
    #                     title=u"登录港口与获救的关联统计", xlabel=u'登录港口', ylabel=u'人数', xresult=True)
    # multiple_feature_result_correlation(train_data, features=[Feature.sex, Feature.p_class],
    #                              title=u'性别与获救的关联统计')
    # feature_result_correlation(train_data, feature=Feature.sib_sp,
    #                     title=u'堂兄弟姐妹个数与获救的关联统计', xlabel=u'堂兄弟姐妹个数', ylabel="人数", xresult=False)
    # feature_result_correlation(train_data, feature=Feature.parch,
    #                     title=u'父母孩子个数与获救的关联统计', xlabel=u'父母孩子个数', ylabel="人数", xresult=False)
    # feature_result_correlation(train_data, feature=Feature.cabin,
    #                     title=u'有无客舱与获救的关联统计', xlabel=u"有无客舱", ylabel=u"人数", xresult=True)
    plt.show()
    print()
