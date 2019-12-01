# -*- coding: utf-8 -*-
# @Time    : 23/11/18 上午10:51
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Titanic.py


import os
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import FeatureUnion
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from matplotlib import pyplot as plt

# 屏蔽警告
import warnings
warnings.filterwarnings("ignore")

TITANIC_PATH = os.path.join("garbage", "titanic")


def load_titanic_data(filename, titanic_path=TITANIC_PATH):
    csv_path = os.path.join(titanic_path, filename)
    return pd.read_csv(csv_path)


train_data = load_titanic_data("train.csv")
test_data = load_titanic_data("test.csv")

# print(train_data.head())
# print(train_data.info())
# print(train_data.describe())

# print(train_data["Survived"].value_counts())
# print(train_data["Pclass"].value_counts())
# print(train_data["Sex"].value_counts())
# print(train_data["Embarked"].value_counts())


class DataFrameSelector(BaseEstimator, TransformerMixin):
    def __init__(self, attributes_names):
        self.attributes_names = attributes_names

    def fit(self, X, y=None):
        return self

    def transform(self, X):
        return X[self.attributes_names]


# 以中位数填充数值类特征缺失值
imputer = SimpleImputer(strategy="median")

num_pipline = Pipeline([
    ("select_numeric", DataFrameSelector(["Age", "SibSp", "Parch", "Fare"])),
    ("imputer", SimpleImputer(strategy="median"))
])

# print(num_pipline.fit_transform(train_data))


# 以众数填充文本类特征缺失值
class MostFrequentImputer(BaseEstimator, TransformerMixin):
    def fit(self, X, Y=None):
        self.most_frequent_ = pd.Series([X[c].value_counts().index[0] for c in X], index=X.columns)
        return self

    def transform(self, X, y=None):
        return X.fillna(self.most_frequent_)


cat_pipeline = Pipeline([
    ("select_cat", DataFrameSelector(["Pclass", "Sex", "Embarked"])),
    ("imputer", MostFrequentImputer()),
    ("cat_encoder", OneHotEncoder(sparse=False))
])

# print(cat_pipeline.fit_transform(train_data))

preprocess_pipeline = FeatureUnion(transformer_list=[
    ("num_pipeline", num_pipline),
    ("cat_pipeline", cat_pipeline)
])

X_train = preprocess_pipeline.fit_transform(train_data)
# print(X_train)

y_train = train_data["Survived"]

# 支持向量机分类起
svm_clf = SVC()
svm_clf.fit(X_train, y_train)
X_test = preprocess_pipeline.fit_transform(test_data)
y_pred = svm_clf.predict(X_test)

svm_scores = cross_val_score(svm_clf, X_train, y_train, cv=10)
# print(svm_scores.mean())

forest_clf = RandomForestClassifier(random_state=42)
forest_scores = cross_val_score(forest_clf, X_train, y_train, cv=10)
# print(forest_scores.mean())

# plt.figure(figsize=(8, 4))
# plt.plot([1]*10, svm_scores, ".")
# plt.plot([2]*10, forest_scores, ".")
# plt.boxplot([svm_scores, forest_scores], labels=("SVM", "Random Forest"))
# plt.ylabel("Accuracy", fontsize=14)
# plt.show()

train_data["AgeBucket"] = train_data["Age"] // 15 * 15
# print(train_data[["AgeBucket", "Survived"]].groupby(['AgeBucket']).mean())

train_data["RelativesOnboard"] = train_data["SibSp"] + train_data["Parch"]
# print(train_data[["RelativesOnboard", "Survived"]].groupby(['RelativesOnboard']).mean())
