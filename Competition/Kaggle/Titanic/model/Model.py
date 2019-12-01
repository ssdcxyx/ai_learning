# -*- coding: utf-8 -*-
# @Time    : 2019-07-04 22:23
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Model.py

from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate
from sklearn.ensemble import BaggingClassifier

import pandas as pd
import numpy as np

from DataPreprocessing import Data
from Setting import FilePath
from Other import File
from DataPreprocessing import Feature
from model import ModelVerification


def linear_model(train_data, test_data, original_test_data, verification=False, cross_val_score=False,
                 plot_learning_curve=False):
    train_df = train_data.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_*|Embarked_*|Sex_*|Pclass_*')
    test_df = test_data.filter(regex='Survived|PassengerId|Age|SibSp|Parch|Fare|Cabin_*|Sex_*|Embarked_*|Pclass_*')

    X_train= train_df.as_matrix()[:, 1:]
    y_train = train_df.as_matrix()[:, 0].astype(int)
    X_test = test_df.as_matrix()[:, 2:]

    # Logistics回归
    clf = LogisticRegression(C=1.0, penalty='l1', tol=1e-6, multi_class='auto', solver='liblinear')
    # 模型系数与特征相关度
    if cross_val_score:
        print(cross_validate(clf, X_train, y_train, cv=5)['test_score'])
    else:
        clf.fit(X_train, y_train)
        predictions = clf.predict(X_test)
        result = pd.DataFrame(
            {'PassengerId': test_df['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
        original_test_data[Feature.survived] = predictions
        File.store_csv(original_test_data, FilePath.logistics_regression_result_feature)
        File.store_csv(result, FilePath.logistics_regression_result)
    if verification:
        print(pd.DataFrame({'columns': list(train_df.columns)[1:], "coef": list(clf.coef_.T)}))
    if plot_learning_curve:
        ModelVerification.plot_learning_curve(clf, u'学习曲线', X_train, y_train, cv=3)
    return clf


def logistics_regression_bagging(train_data, test_data, original_test_data, verification=False, cross_val_score=False,
                 plot_learning_curve=False):
    train_df = train_data.filter(regex='Survived|Age|SibSp|Parch|Fare|Cabin_*|Embarked_*|Sex_*|Pclass_*')
    test_df = test_data.filter(regex='Survived|PassengerId|Age|SibSp|Parch|Fare|Cabin_*|Sex_*|Embarked_*|Pclass_*')

    X_train = train_df.as_matrix()[:, 1:]
    y_train = train_df.as_matrix()[:, 0].astype(int)
    X_test = test_df.as_matrix()[:, 2:]

    clf = LogisticRegression(C=1.0, penalty="l1", tol=1e-6, solver="liblinear")
    bagging_clf = BaggingClassifier(clf, n_estimators=20, max_samples=0.8, max_features=1.0, bootstrap=True,
                                    bootstrap_features=True)
    bagging_clf.fit(X_train, y_train)
    predictions = bagging_clf.predict(X_test)
    result = pd.DataFrame(
        {'PassengerId': test_df['PassengerId'].as_matrix(), 'Survived': predictions.astype(np.int32)})
    original_test_data[Feature.survived] = predictions
    File.store_csv(original_test_data, FilePath.logistics_regression_bagging_result_feature)
    File.store_csv(result, FilePath.logistics_regression_bagging_result)
    if plot_learning_curve:
        ModelVerification.plot_learning_curve(clf, u'学习曲线', X_train, y_train, cv=3)
    return bagging_clf


if __name__ == '__main__':
    handled_train_data = Data.get_data(FilePath.handled_train_data)
    handled_test_data = Data.get_data(FilePath.handled_test_data)
    original_test_data = Data.get_data(FilePath.original_test_data)
    linear_model(handled_train_data, handled_test_data, original_test_data, verification=False, cross_val_score=False,
                 plot_learning_curve=True)
    # logistics_regression_bagging(handled_train_data, handled_test_data, original_test_data, plot_learning_curve=True)
    print()
