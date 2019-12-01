# -*- coding: utf-8 -*-
# @Time    : 2019-06-29 12:25
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : FeatureEngineering.py

from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer

import numpy as np

from DataPreprocessing import Feature
from DataPreprocessing import Data
from Setting import FilePath
from Other import File


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        return X[self.attributes_names]


class MyLabelEncoder(TransformerMixin):

    def __init__(self, *args, **kwargs):
        self.encoder = OneHotEncoder(categories='auto', *args, **kwargs)

    def fit(self, x, y=None):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=None):
        return self.encoder.transform(x), self.encoder.categories_


# 标签数值化
def label_quantization(data, columns):
    pipline = Pipeline([
        ("select_columns", DataFrameSelector(columns)),
        ('label_encoder', MyLabelEncoder())
    ])
    process_data, categories = pipline.fit_transform(data.astype(str))
    process_data = process_data.toarray()
    shape = process_data.shape
    data = data.drop(columns, axis=1)
    for i in range(shape[1]):
        data[columns[0] + '_' + str(categories[0][i])] = process_data[:, i:i + 1]
    return data


# 数值归一化
def number_normal(data, columns):
    pipline = Pipeline([
        ("select_columns", DataFrameSelector(columns)),
        ('label_encoder', StandardScaler())
    ])
    process_data = pipline.fit_transform(data)
    data[columns] = process_data
    return data


def data_padding(data, columns, strategy='most_frequent'):
    pipline = Pipeline([
        ("select_columns", DataFrameSelector(columns)),
        ('label_encoder', SimpleImputer(strategy=strategy))
    ])
    process_data = pipline.fit_transform(data)
    data[columns] = process_data
    return data


# 填充缺失的年龄属性
def padding_missing_ages(data):
    numerical_data = data.filter(regex='Age|SibSp|Parch|Fare|Cabin_*|Embarked_*|Pclass_*')
    known_age = numerical_data[numerical_data[Feature.age].notnull()].as_matrix()
    unknown_age = numerical_data[numerical_data[Feature.age].isnull()].as_matrix()

    y = known_age[:, 0]
    X = known_age[:, 1:]

    model = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=1)
    model.fit(X, y)

    predict = model.predict(unknown_age[:, 1:])
    data.loc[(data[Feature.age].isnull()), Feature.age] = predict
    return data, model


# 判断有无客舱
def with_cabin(df):
    df.loc[(df.Cabin.notnull()), 'Cabin'] = "Yes"
    df.loc[(df.Cabin.isnull()), 'Cabin'] = "No"
    return df


def data_process(data):
    # 填充Fare缺失的数据
    handled_data = data_padding(data, [Feature.fare])
    handled_data = with_cabin(handled_data)
    handled_data = label_quantization(handled_data, [Feature.cabin])
    handled_data = label_quantization(handled_data, [Feature.sex])
    handled_data = label_quantization(handled_data, [Feature.p_class])
    handled_data = label_quantization(handled_data, [Feature.embarked])
    handled_data = number_normal(handled_data, [Feature.fare])
    handled_data, _ = padding_missing_ages(handled_data)
    handled_data = number_normal(handled_data, [Feature.age])
    return handled_data


if __name__ == '__main__':
    original_train_data = Data.get_data(FilePath.original_train_data)
    original_test_data = Data.get_data(FilePath.original_test_data)
    # 剔除训练数据中embarked为NaN
    original_train_data = original_train_data.dropna(subset=[Feature.embarked])
    train_data_len = original_train_data.shape[0]
    data = original_train_data.append(original_test_data)[[Feature.survived,
                                                           Feature.passenger_id, Feature.p_class, Feature.name,
                                                           Feature.sex, Feature.age, Feature.sib_sp, Feature.parch,
                                                           Feature.ticket, Feature.fare, Feature.cabin,
                                                           Feature.embarked]]

    handled_data = data_process(data)

    handled_train_data = handled_data[:train_data_len]
    handled_test_data = handled_data[train_data_len:]

    File.store_csv(handled_train_data, FilePath.handled_train_data)
    File.store_csv(handled_test_data, FilePath.handled_test_data)

