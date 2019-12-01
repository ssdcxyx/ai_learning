# -*- coding: utf-8 -*-
# @Time    : 2019-07-05 21:24
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : ModelVerification.py

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import learning_curve

from DataPreprocessing import Feature
from Setting import FilePath
from DataPreprocessing import Data
from Other import File


def classification_bad_cases(original_train_data, handled_train_data, estimator):
    split_train, split_cv = train_test_split(handled_train_data, test_size=0.3, random_state=0)
    train_df = split_train.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    clf = estimator(C=1.0, penalty="l1", tol=1e-6, solver='liblinear')
    clf.fit(train_df.as_matrix()[:, 1:], train_df.as_matrix()[:, 0])
    cv_df = split_cv.filter(regex='Survived|Age_.*|SibSp|Parch|Fare_.*|Cabin_.*|Embarked_.*|Sex_.*|Pclass_.*')
    predictions = clf.predict(cv_df.as_matrix()[:, 1:])
    bad_cases = original_train_data.loc[
        original_train_data['PassengerId'].isin(split_cv[predictions != cv_df.as_matrix()[:, 0]]['PassengerId'].values)]
    print(bad_cases)
    File.store_csv(bad_cases, FilePath.logistics_regression_classification_bad_cases)


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None, n_jobs=1, train_sizes=np.linspace(.05, 1., 20),
                        verbose=0, plot=True):
    train_sizes, train_scores, test_scores = learning_curve(estimator, X, y, cv=cv, n_jobs=n_jobs,
                                                            train_sizes=train_sizes, verbose=verbose)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    if plot:
        plt.figure()
        plt.title(title)
        if ylim is not None:
            plt.ylim(*ylim)
        plt.xlabel(u'训练样本数')
        plt.ylabel(u"得分")
        plt.grid()

        plt.fill_between(train_sizes, train_scores_mean - train_scores_std, train_scores_mean + train_scores_std)
        plt.fill_between(train_sizes, test_scores_mean - test_scores_std, test_scores_mean + test_scores_std)

        plt.plot(train_sizes, train_scores_mean, 'o-', color="b", label=u"训练集上得分")
        plt.plot(train_sizes, test_scores_mean, 'o-', color="r", label=u"交叉验证集上得分")

        plt.legend(loc="best")

        plt.draw()
        plt.show()
    midpoint = ((train_scores_mean[-1] + train_scores_std[-1]) + (test_scores_mean[-1] - test_scores_std[-1])) / 2
    diff = (train_scores_mean[-1] + train_scores_std[-1]) - (test_scores_mean[-1] - test_scores_std[-1])
    return midpoint, diff


if __name__ == '__main__':
    handled_train_data = Data.get_data(FilePath.handled_train_data)
    handled_test_data = Data.get_data(FilePath.handled_test_data)
    original_train_data = Data.get_data(FilePath.original_train_data)
    classification_bad_cases(original_train_data, handled_train_data, LogisticRegression)
    print()
