# -*- coding: utf-8 -*-
# @Time    : 2019-05-10 10:43
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : DataProcess.py

from sklearn.pipeline import Pipeline
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

import pandas as pd

from handle import GetData
from setting import Constant
from handle import File
from setting import tool


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attributes_names]


class MyLabelEncoder(TransformerMixin):

    def __init__(self, *args, **kwargs):
        self.encoder = LabelEncoder(*args, **kwargs)

    def fit(self, x, y=0):
        self.encoder.fit(x)
        return self

    def transform(self, x, y=0):
        return self.encoder.transform(x)


def label_quantization(data, columns):
    pipline = Pipeline([
        ("select_columns", DataFrameSelector(columns)),
        ('label_encoder', MyLabelEncoder())
    ])
    process_data = pipline.fit_transform(data)
    data[columns] = process_data
    return data


# 城市标签数值化
def city_label_quantization(data, columns):
    city_quantization_df = pd.DataFrame()
    pipline = Pipeline([
        ("select_columns", DataFrameSelector(columns)),
        ('label_encoder', OneHotEncoder())
    ])
    process_data = pipline.fit_transform(data).toarray()
    for i in range(0, len(process_data)):
        city_quantization_df = city_quantization_df.append(
            pd.DataFrame({"city1": process_data[i][0], "city2": process_data[i][1], "city3": process_data[i][2],
                          "city4": process_data[i][3], "city5": process_data[i][4], "city6": process_data[i][5],
                          "city7": process_data[i][6]}, index={data.index[i]})
        )
    data = data.drop(['city'], axis=1)
    data = pd.concat([data, city_quantization_df], axis=1)
    return data


# 星座数值化
def constellation_quantization(data):
    constellations = ['白羊座', '金牛座', '双子座', '巨蟹座', '狮子座', '处女座',
                      '天秤座', '天蝎座', '射手座', '摩羯座', '水瓶座', '双鱼座']
    constellation_quantization_df = pd.DataFrame()
    for index, row in data.iterrows():
        result = tool.twevle_quantile_of_unit_circle(constellations.index(row['constellation']))
        constellation_quantization_df = constellation_quantization_df.append(
            pd.DataFrame({'constellation_x': result[0], 'constellation_y': result[1]}, index={index}))
    data = data.drop(['constellation'], axis=1)
    data = pd.concat([data, constellation_quantization_df], axis=1)
    File.store_csv(data, file_path=Constant.user_profile_handle_table)
    return data


# 缺失值填充
def fill_missing_value(data, columns, strategy="constant", fill_value=0):
    pipline = Pipeline([
        ("select_columns", DataFrameSelector(columns)),
        ('label_encoder', SimpleImputer(strategy=strategy, fill_value=fill_value))
    ])
    process_data = pipline.fit_transform(data)
    data[columns] = process_data
    return data


# 数值归一化
def number_normal(data):
    scaler = StandardScaler()
    process_data = scaler.fit_transform(data)
    return process_data


# 剔除交易记录中申购和赎回全部为零或空的行
def wipe_empty_data(data):
    data = fill_missing_value(data, columns=['category1', 'category2', 'category3', 'category4'])
    print("original img(balance) length:", len(data))
    data = data[(data['total_purchase_amt'] != 0) | (data['total_redeem_amt'] != 0)]
    print("after wipe:", len(data))
    return data.sort_values(by="user_id", ascending=True)


if __name__ == "__main__":
    File.store_csv(wipe_empty_data(GetData.get_data(Constant.user_balance_table)),
                   file_path=Constant.fill_missing_value_wipe_empty_balance_table)
    print()

