# -*- coding: utf-8 -*-
# @Time    : 16/11/18 下午7:03
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : HousingForecast.py

import os
import tarfile
from six.moves import urllib

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import hashlib

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedShuffleSplit
from pandas.plotting import scatter_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OrdinalEncoder
from sklearn.preprocessing import OneHotEncoder
# from sklearn.preprocessing import CategoricalEncoder
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion
from sklearn.compose import ColumnTransformer

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.externals import joblib
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import randint
from sklearn.svm import SVR
from scipy.stats import expon, reciprocal


DOWNLOAD_ROOT = "https://raw.githubusercontent.com/ageron/handson-ml/master/"
HOUSING_PATH = "garbage/housing"
HOUSING_URL = DOWNLOAD_ROOT + HOUSING_PATH + "/housing.tgz"


def fetch_housing_data(housing_url=HOUSING_URL, housing_path=HOUSING_PATH):
    # 获取并下载数据
    if not os.path.isdir(housing_path):
        os.makedirs(housing_path)
    tgz_path = os.path.join(housing_path, "housing.tgz")
    urllib.request.urlretrieve(housing_url, tgz_path)
    housing_tgz = tarfile.open(tgz_path)
    housing_tgz.extractall(path=housing_path)
    housing_tgz.close()


def load_housing_data(housing_path=HOUSING_PATH):
    # 加载数据
    csv_path = os.path.join(housing_path, "housing_price.csv")
    return pd.read_csv(csv_path)


fetch_housing_data()
housing = load_housing_data()
# print(housing.head())
# print(housing.info())
# print(housing["ocean_proximity"].value_counts())
# print(housing.describe())

# housing.hist(bins=50, figsize=(20, 15))
# plt.show()


# def split_train_test(img, test_ratio):
#     # 随机打乱数据
#     shuffled_indices = np.random.permutation(len(img))
#     test_set_size = int(len(img) * test_ratio)
#     test_indices = shuffled_indices[:test_set_size]
#     train_indices = shuffled_indices[test_set_size:]
#     return img.iloc[train_indices], img.iloc[test_indices]
#
#
# train_set, test_set = split_train_test(housing, 0.2)
# print(len(train_set), "train +", len(test_set), "test")


# def test_set_check(identifier, test_ratio, hash):
#     return hash(np.int64(identifier)).digest()[-1] < 256 * test_ratio
#
#
# def split_train_test_by_id(img, test_ratio, id_columns, hash=hashlib.md5):
#     ids = img[id_columns]
#     in_test_set = ids.apply(lambda id_: test_set_check(id_, test_ratio, hash))
#     return img.loc[~in_test_set], img.loc[in_test_set]


# # 加入行索作为列"index"
# housing_with_id = housing.reset_index()
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "index")

# housing_with_id = housing["longitude"] * 1000 + housing["latitude"]
# train_set, test_set = split_train_test_by_id(housing_with_id, 0.2, "id")

# train_set, test_set = train_test_split(housing, test_size=0.2, random_state=42)

housing["income_cat"] = np.ceil(housing["median_income"] / 1.5)
housing['income_cat'].where(housing["income_cat"] < 5, 5.0, inplace=True)

split = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)

for train_index, test_index in split.split(housing, housing["income_cat"]):
    start_train_set = housing.loc[train_index]
    start_test_set = housing.loc[test_index]

# print(start_test_set["income_cat"].value_counts()/len(start_test_set))

# print(housing["income_cat"].value_counts()/len(housing))

# 使数据回到初始状态
for set in (start_train_set, start_test_set):
    set.drop(["income_cat"], axis=1, inplace=True)

# housing = start_train_set.copy()

# 绘制散点图
# housing.plot(kind="scatter", x="longitude", y="latitude", alpha=0.4,
#              s=housing["population"]/100, label="population",
#              c="median_house_value", cmap=plt.get_cmap("jet"), colorbar=True)
# plt.show()

# 相关系数
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=True))

# 属性相关
# attributes = ["median_house_value", "median_income", "total_rooms", "housing_median_age"]
# scatter_matrix(housing[attributes], figsize=(12, 8))
# plt.show()

# housing.plot(kind="scatter", x="median_income", y="median_house_value", alpha=0.1)
# plt.show()

# 特征组合
# housing["rooms_per_household"] = housing["total_rooms"]/housing["households"]
# housing["bedrooms_per_room"] = housing["total_bedrooms"]/housing["total_rooms"]
# housing["population_per_household"] = housing["population"]/housing["households"]
#
# corr_matrix = housing.corr()
# print(corr_matrix["median_house_value"].sort_values(ascending=False))

housing = start_train_set.drop("median_house_value", axis=1)
housing_labels = start_train_set["median_house_value"].copy()

# total_bedrooms属性缺失处理
# housing.dropna(subset=["total_bedrooms"])  # 去除对应街区
# housing.drop("total_bedrooms", axis=1)  # 去除整个属性
# median = housing["total_bedrooms"].median()
# housing["total_bedrooms"].fillna(median)  # 进行赋值(中位数)

# Imputer 处理缺失值
# imputer = SimpleImputer(strategy="median")

housing_num = housing.drop("ocean_proximity", axis=1)
# imputer.fit(housing_num)
# print(imputer.statistics_)
# print(housing_num.median().values)

# X = imputer.transform(housing_num)
# housing_tr = pd.DataFrame(X, columns=housing_num.columns)

# housing_cat = housing["ocean_proximity"]

# 文本标签特征列转换
# LabelEncoder
# encoder = LabelEncoder()
# housing_cat_encoded = encoder.fit_transform(housing_cat)
# print(housing_cat_encoded)
# housing_cat_encoded, housing_categories = housing_cat.factorize()
# print(housing[:10])
# print(encoder.classes_)

# OrdinalEncoder
# encoder = OrdinalEncoder()
# housing_cat = housing["ocean_proximity"]
# housing_cat_encoded = encoder.fit_transform(housing_cat)

# housing_cat_reshape = housing_cat.values.reshape(-1, 1)

# OneHotEncoder
# encoder = OneHotEncoder(sparse=False)
# housing_cat_1hot = encoder.fit_transform(housing_cat_reshape)
# print(housing_cat_1hot)
# print(encoder.categories_)

# 自定义转换器
rooms_ix, bedrooms_ix, population_ix, household_ix = 3, 4, 5, 6


class CombinedAttributesAdder(BaseEstimator, TransformerMixin):

    def __init__(self, add_bedrooms_per_room=True):
        self.add_bedrooms_per_room = add_bedrooms_per_room

    def fit(self, X, y=None):
        return self

    def transform(self, X, y=None):
        rooms_per_household = X[:, rooms_ix] / X[:, household_ix]
        population_per_household = X[:, population_ix] / X[:, household_ix]
        if self.add_bedrooms_per_room:
            bedrooms_per_room = X[:, bedrooms_ix] / X[:, rooms_ix]
            return np.c_[X, rooms_per_household, population_per_household, bedrooms_per_room]
        else:
            return np.c_[X, rooms_per_household, population_per_household]


# attr_adder = CombinedAttributesAdder(add_bedrooms_per_room=False)
# housing_extra_attribs = attr_adder.transform(housing.values)

# 转换流水线
num_pipeline = Pipeline([
    ('imputer', SimpleImputer(strategy="median")),
    ('attribs_adder', CombinedAttributesAdder()),
    ('std_scaler', StandardScaler())
])
# housing_num_tr = num_pipeline.fit_transform(housing_num)


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attribute_names):
        self.attribute_names = attribute_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attribute_names].values


num_attribs = list(housing_num)
cat_attribs = ["ocean_proximity"]

# num_pipeline = Pipeline([
#     ('selector', DataFrameSelector(num_attribs)),
#     ('imputer', SimpleImputer(strategy="median")),
#     ('attribs_adder', CombinedAttributesAdder()),
#     ('std_scaler', StandardScaler())
# ])
#
# cat_pipeline = Pipeline([
#     ('selector', DataFrameSelector(cat_attribs)),
#     ('cat_encoder', CategoricalEncoder(encoding="onehot-dense"))
# ])
#
# full_pipeline = FeatureUnion(transformer_list=[
#     ("num_pipeline", num_pipeline),
#     ("cat_pipeline", cat_pipeline)
# ])
#
# housing_prepared = full_pipeline.fit_transform(housing)
# print(housing_prepared.shape)

full_pipeline = ColumnTransformer([
    ("num", num_pipeline, num_attribs),
    ("cat", OneHotEncoder(), cat_attribs)
])
housing_prepared = full_pipeline.fit_transform(housing)

lin_reg = LinearRegression()
lin_reg.fit(housing_prepared, housing_labels)
# some_data = housing.iloc[:5]
# some_labels = housing_labels.iloc[:5]
# some_data_prepared = full_pipeline.transform(some_data)
# print("Prediction:\t", lin_reg.predict(some_data_prepared))
# print("Labels:\t\t", list(some_labels))
# housing_predictions = lin_reg.predict(housing_prepared)
# line_mse = mean_squared_error(housing_labels, housing_predictions)
# line_mse = np.sqrt(line_mse)
# print(line_mse)
# 效用函数
line_scores = cross_val_score(lin_reg, housing_prepared, housing_labels,
                            scoring="neg_mean_squared_error", cv=10)
line_rmse_scores = np.sqrt(-line_scores)

tree_reg = DecisionTreeRegressor()
tree_reg.fit(housing_prepared, housing_labels)
# housing_predictions = tree_reg.predict(housing_prepared)
# tree_mse = mean_squared_error(housing_labels, housing_predictions)
# tree_mse = np.sqrt(tree_mse)
# print(tree_mse)
tree_scores = cross_val_score(tree_reg, housing_prepared, housing_labels,
                         scoring="neg_mean_squared_error", cv=10)
tree_rmse_scores = np.sqrt(-tree_scores)

forest_reg = RandomForestRegressor()
forest_reg.fit(housing_prepared, housing_labels)
forest_scores =cross_val_score(forest_reg, housing_prepared, housing_labels,
                               scoring="neg_mean_squared_error", cv=10)
forest_rmse_scores = np.sqrt(-forest_scores)


def display_scores(scores):
    print("Scores:",  scores)
    print("Mean:", scores.mean())
    print("Standard deviation:", scores.std())


# display_scores(tree_rmse_scores)
# display_scores(line_rmse_scores)
# display_scores(forest_rmse_scores)

# 保存模型
# joblib.dump(forest_reg, "model/forest_reg.pkl")
# forest_reg_loaded = joblib.load("model/forest_reg.pkl")

# 网络搜索
param_grid = [
    {'n_estimators': [3, 10, 30], 'max_features': [2, 4, 6, 8]},
    {'bootstrap': [False], 'n_estimators':[3, 10], 'max_features':[2, 3, 4]},
]

grid_search = GridSearchCV(forest_reg, param_grid, cv=4,
                           scoring="neg_mean_squared_error")
grid_search.fit(housing_prepared, housing_labels)

# print(grid_search.best_params_)
# print(grid_search.best_estimator_)
#
# cvres = grid_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)

# 随机搜索
param_distribs = {
    'n_estimators': randint(low=1, high=200),
    'max_features': randint(low=1, high=8)
}
#
# forest_reg = RandomForestRegressor(random_state=42)
rnd_search = RandomizedSearchCV(forest_reg, param_distributions=param_distribs,
                                n_iter=10, cv=5, scoring="neg_mean_squared_error", random_state=42)
rnd_search.fit(housing_prepared, housing_labels)

# cvres = rnd_search.cv_results_
# for mean_score, params in zip(cvres["mean_test_score"], cvres["params"]):
#     print(np.sqrt(-mean_score), params)

feature_importances = grid_search.best_estimator_.feature_importances_
# print(feature_importances)

extra_attribs = ["rooms_per_hhold", "pop_per_hhold", "bedrooms_per_room"]
cat_encoder = full_pipeline.named_transformers_["cat"]
cat_one_hot_attribs = list(cat_encoder.categories_[0])
attributes = num_attribs + extra_attribs + cat_one_hot_attribs
# print(sorted(zip(feature_importances, attributes), reverse=True))
#
# final_model = grid_search.best_estimator_
#
# X_test = start_test_set.drop("median_house_value", axis=1)
# y_test = start_test_set["median_house_value"].copy()
#
# X_test_prepared = full_pipeline.transform(X_test)
#
# final_predictions = final_model.predict(X_test_prepared)
#
# final_mse = mean_squared_error(y_test, final_predictions)
# final_rmse = np.sqrt(final_mse)
# print(final_rmse)

# SVM
# param_grid = [
#     {'kernel': ['linear'], 'C':[10., 30., 100., 300., 1000., 3000., 10000., 30000.0]},
#     {'kernel': ['rbf'], 'C':[1.0, 3.0, 10., 30., 100., 300., 1000.0],
#      'gamma': [0.01, 0.03, 0.1, 0.3, 1.0, 3.0]},
# ]
# svm_reg = SVR()
# grid_search = GridSearchCV(svm_reg, param_grid, cv=5,
#                            scoring='neg_mean_squared_error', verbose=2, n_jobs=4)
# grid_search.fit(housing_prepared, housing_labels)

# param_distribs = {
#     'kernel': ['linear', 'rbf'],
#     'C': reciprocal(20, 200000),
#     'gamma': expon(scale=1.0)
# }
# svm_reg = SVR()
# rnd_search = RandomizedSearchCV(svm_reg, param_distributions=param_distribs,
#                                 n_iter=50, cv=5, scoring='neg_mean_squared_error',
#                                 verbose=2, n_jobs=4, random_state=42)
# rnd_search.fit(housing_prepared, housing_labels)
#
# negative_mse = rnd_search.best_score_
# rmse = np.sqrt(-negative_mse)
# print(rmse)
# print(rnd_search.best_params_)

# expon_distrib = expon(scale=1.)
# samples = expon_distrib.rvs(10000, random_state=42)
# plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.title("Exponential distribution (scale=1.0)")
# plt.hist(samples, bins=50)
# plt.subplot(122)
# plt.title("Log of this distribution")
# plt.hist(np.log(samples), bins=50)
# plt.show()

# reciprocal_distrib = reciprocal(20, 200000)
# samples = reciprocal_distrib.rvs(10000, random_state=42)
# plt.figure(figsize=(10, 4))
# plt.subplot(121)
# plt.title("Reciprocal distribution (scale=1.0)")
# plt.hist(samples, bins=50)
# plt.subplot(122)
# plt.title("Log of this distribution")
# plt.hist(np.log(samples), bins=50)
# plt.show()


def indices_of_top_k(arr, k):
    test = np.array(arr)
    return np.sort(np.argpartition(np.array(arr), -k)[-k:])


class TopFeatureSelector(BaseEstimator, TransformerMixin):
    def __init__(self, feature_importances, k):
        self.feature_importance = feature_importances
        self.k = k

    def fit(self, X, y=None):
        self.feature_indices = indices_of_top_k(self.feature_importance, self.k)
        return self

    def transform(self, X):
        return X[:, self.feature_indices]


k = 5
# top_k_feature_indices = indices_of_top_k(feature_importances, k)
# print(top_k_feature_indices)
# print(np.array(attributes)[top_k_feature_indices])
# print(sorted(zip(feature_importances, attributes), reverse=True)[:k])

preparation_and_feature_selection_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k))
])
housing_prepared_top_k_feature = preparation_and_feature_selection_pipeline.fit_transform(housing)
# print(housing_prepared_top_k_feature[0:3])

prepare_select_and_predict_pipeline = Pipeline([
    ('preparation', full_pipeline),
    ('feature_selection', TopFeatureSelector(feature_importances, k)),
    ('forest_reg', RandomForestRegressor(**rnd_search.best_params_))
])
preparation_and_feature_selection_pipeline.fit(housing, housing_labels)

# some_data = housing.iloc[:4]
# some_labels = housing_labels.iloc[:4]
# print("Predictions:\t", preparation_and_feature_selection_pipeline.predict(some_data))
# print("Labels:\t\t", list(some_labels))

# param_grid = [
#     {'preparation_num_imputer_strategy': ['mean', 'median', 'most_frequent'],
#      'feature_selection_k': list(range(1, len(feature_importances) + 1))}
# ]
#
# grid_search_prep = GridSearchCV(preparation_and_feature_selection_pipeline, param_grid, cv=5,
#                                 scoring="neg_mean_squared_error", verbose=2, n_jobs=4)
# grid_search_prep.fit(housing, housing_labels)
#
