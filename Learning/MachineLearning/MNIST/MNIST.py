# -*- coding: utf-8 -*-
# @Time    : 20/11/18 下午6:18
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : MNIST.py


from sklearn.datasets import fetch_mldata
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.base import clone
from sklearn.base import BaseEstimator
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import cross_val_predict
from sklearn.metrics import confusion_matrix
from sklearn.metrics import precision_score, recall_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.multiclass import OneVsOneClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.dummy import DummyClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score
from scipy.ndimage.interpolation import shift

# 屏蔽警告
import warnings
warnings.filterwarnings("ignore")

mnist = fetch_mldata('MNIST original')

X, y = mnist["img"], mnist["target"]
some_digit = X[36000]
# some_digit_image = some_digit.reshape(28, 28)
# plt.imshow(some_digit_image, cmap=matplotlib.cm.binary, interpolation="nearest")
# plt.axis("off")
# plt.show()

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]
# 随机打乱
shuffle_index = np.random.permutation(60000)
X_train, y_train = X_train[shuffle_index], y_train[shuffle_index]

y_train_5 = (y_train == 5)
y_test_5 = (y_test == 5)

# 随机梯度下降分类器
sgd_clf = SGDClassifier(random_state=42)
sgd_clf.fit(X_train, y_train_5)
# print(sgd_clf.predict([some_digit]))

# 交叉验证
# skfolds = StratifiedKFold(n_splits=3, random_state=42)
# for train_index, test_index in skfolds.split(X_train, y_train_5):
#     clone_clf = clone(sgd_clf)
#     X_train_folds = X_train[train_index]
#     y_train_folds = (y_train_5[train_index])
#     X_test_fold = X_train[test_index]
#     y_test_fold = (y_train_5[test_index])
#     clone_clf.fit(X_train_folds, y_train_folds)
#     y_pred = clone_clf.predict(X_test_fold)
#     n_correct = sum(y_pred == y_test_fold)
#     print(n_correct / len(y_pred))


# class Never5Classifier(BaseEstimator):
#     def fit(self, X, y=None):
#         pass
#
#     def predict(self, X):
#         return np.zeros((len(X), 1), dtype=bool)
#
#
# never_5_clf = Never5Classifier()
# cross_val_score(never_5_clf, X_train, y_train_5, cv=3, scoring="accuracy")

y_train_pred = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3)
# 混淆矩阵
# print(confusion_matrix(y_train_5, y_train_pred))

# 准确率
# print(precision_score(y_train_5, y_train_pred))
# 召回率
# print(recall_score(y_train_5, y_train_pred))
# 调和平均值
# print(f1_score(y_train_5, y_train_pred))

y_scores = cross_val_predict(sgd_clf, X_train, y_train_5, cv=3, method="decision_function")
precisions, recalls, thresholds = precision_recall_curve(y_train_5, y_scores)


def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
    plt.plot(thresholds, precisions[:-1], "b--", label="Precision")
    plt.plot(thresholds, recalls[:-1], "g--", label="Recall")
    plt.xlabel("Threshold")
    plt.legend(loc="upper left")
    plt.ylim([0, 1])


# plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
# plt.show()

y_train_pred_90 = (y_scores > 70000)
# print(precision_score(y_train_5, y_train_pred_90))
# print(recall_score(y_train_5, y_train_pred_90))


def plot_precision_vs_recall(precisions, recalls):
    plt.plot(recalls, precisions, 'b--', label="Precision")
    plt.xlabel("Recall", fontsize=16)
    plt.ylabel("Precision", fontsize=16)
    plt.axis([0, 1, 0, 1])


# plt.figure(figsize=(8, 6))
# plot_precision_vs_recall(precisions, recalls)
# plt.show()

# ROC曲线
fpr, tpr, thresholds = roc_curve(y_train_5, y_scores)


def plot_roc_curve(fpr, tpr, label=None):
    plt.plot(fpr, tpr, linewidth=2, label=label)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate')
    plt.ylabel("True Positive Rate")
    plt.show()


# plot_roc_curve(fpr, tpr)

# print(roc_auc_score(y_train_5, y_scores))

forest_clf = RandomForestClassifier(random_state=42)
y_probas_forest = cross_val_predict(forest_clf, X_train, y_train_5, cv=3, method="predict_proba")

y_scores_forest = y_probas_forest[:, 1]
fpr_forest, tpr_forest, thresholds_forest = roc_curve(y_train_5, y_scores_forest)

# plt.plot(fpr, tpr, "b:", label="SGD")
# plot_roc_curve(fpr_forest, tpr_forest, "Random Forest")
# plt.legend(loc="bottom right")
# plt.show()

# print(roc_auc_score(y_train_5, y_scores_forest))

# 多类分类
sgd_clf.fit(X_train, y_train)
# print(sgd_clf.predict([some_digit]))

some_digit_score = sgd_clf.decision_function([some_digit])
# print(some_digit_score)

# OvO策略
# ovo_clf = OneVsOneClassifier(SGDClassifier(random_state=42))
# ovo_clf.fit(X_train, y_train)
# print(ovo_clf.predict([some_digit]))

forest_clf.fit(X_train, y_train)
# print(forest_clf.predict([some_digit]))
# print(forest_clf.predict_proba([some_digit]))

# print(cross_val_score(sgd_clf, X_train, y_train, cv=3, scoring="accuracy"))

# 输入正则化
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train.astype(np.float64))
# print(cross_val_score(sgd_clf, X_train_scaled, y_train, cv=3, scoring="accuracy"))

# 混淆矩阵
y_train_pred = cross_val_predict(sgd_clf, X_train_scaled, y_train, cv=3)
conf_mx = confusion_matrix(y_train, y_train_pred)
# print(conf_mx)

# plt.matshow(conf_mx, cmap=plt.cm.gray)
# plt.show()

# keepdims保持二维性
row_sums = conf_mx.sum(axis=1, keepdims=True)
norm_conf_mx = conf_mx / row_sums

np.fill_diagonal(norm_conf_mx, 0)
# plt.matshow(norm_conf_mx, cmap=plt.cm.gray)
# plt.show()


def plot_digit(data):
    image = data.reshape(28, 28)
    plt.imshow(image, cmap=matplotlib.cm.binary,
               interpolation="nearest")
    plt.axis("off")


def plot_digits(instances, images_per_row=10, **options):
    size = 28
    images_per_row = min(len(instances), images_per_row)
    images = [instance.reshape(size, size) for instance in instances]
    n_rows = (len(instances) - 1) // images_per_row + 1
    row_images = []
    # n_empty = n_rows * images_per_row - len(instances)
    # images.append(np.zeros((size, size*n_empty)))
    for row in range(n_rows):
        rimages = images[row * images_per_row: (row+1) * images_per_row]
        row_images.append(np.concatenate(rimages, axis=1))
    image = np.concatenate(row_images, axis=0)
    plt.imshow(image, cmap=matplotlib.cm.binary, **options)
    plt.axis("off")


cl_a, cl_b = 3, 5
X_aa = X_train[(y_train == cl_a) & (y_train_pred == cl_a)]
X_ab = X_train[(y_train == cl_a) & (y_train_pred == cl_b)]
X_ba = X_train[(y_train == cl_b) & (y_train_pred == cl_a)]
X_bb = X_train[(y_train == cl_b) & (y_train_pred == cl_b)]
# plt.figure(figsize=(8, 8))
# plt.subplot(221); plot_digits(X_aa[:25], images_per_row=5)
# plt.subplot(222); plot_digits(X_ab[:25], images_per_row=5)
# plt.subplot(223); plot_digits(X_ba[:25], images_per_row=5)
# plt.subplot(224); plot_digits(X_bb[:25], images_per_row=5)
# plt.show()

# 多输出分类器
y_train_large = (y_train >= 7)
y_train_odd = (y_train % 2 == 1)
y_multilabel = np.c_[y_train_large, y_train_odd]

# knn_clf = KNeighborsClassifier()
# knn_clf.fit(X_train, y_multilabel)
#
# print(knn_clf.predict([some_digit]))

noise = np.random.randint(0, 100, (len(X_train), 784))
X_train_mod = X_train + noise
noise = np.random.randint(0, 100, (len(X_test), 784))
X_test_mod = X_test + noise
y_train_mod = X_train
y_test_mod = X_test

some_index = 5500

# plt.subplot(121); plot_digit(X_test_mod[some_index])
# plt.subplot(122); plot_digit(y_test_mod[some_index])
# plt.show()

# knn_clf.fit(X_train_mod, y_train_mod)
# clean_digit = knn_clf.predict([X_test_mod[some_index]])
# plot_digit(clean_digit)
# plt.show()

# 随机分类器
# dmy_clf = DummyClassifier()
# y_probas_dmy = cross_val_predict(dmy_clf, X_train, y_train_5, cv=3, method="predict_proba")
# y_scores_dmy = y_probas_dmy[:, 1]
#
# fprr, tprr, thresholdsr = roc_curve(y_train_5, y_scores_dmy)
# plot_roc_curve(fprr, tprr)

# 表格搜索
param_grid = [{"weights": ["distance"], 'n_neighbors':[4]}]
knn_clf = KNeighborsClassifier()
grid_search = GridSearchCV(knn_clf, param_grid, cv=2, verbose=3, n_jobs=-1)
grid_search.fit(X_train, y_train)
#
# print(grid_search.best_params_)
# print(grid_search.best_score_)
# y_pred = grid_search.predict(X_test)
# print(accuracy_score(y_test, y_pred))


# 移动图片
def shift_image(image, dx, dy):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [dy, dx], cval=0, mode="constant")
    return shifted_image.reshape([-1])


X_train_augmented = [image for image in X_train]
y_train_augmented = [label for label in y_train]

for dx, dy in [(0, 1), (-1, 0), (0, 1), (0, -1)]:
    for image, label in zip(X_train, y_train):
        X_train_augmented.append(shift_image(image, dx, dy))
        y_train_augmented.append(label)


X_train_augmented = np.array(X_train_augmented)
y_train_augmented = np.array(y_train_augmented)

shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = X_train_augmented[shuffle_idx]
y_train_augmented = y_train_augmented[shuffle_idx]

knn_clf = KNeighborsClassifier(**grid_search.best_params_)
knn_clf.fit(X_train_augmented, y_train_augmented)
y_pred = knn_clf.predict(X_test)
print(accuracy_score(y_test, y_pred))
