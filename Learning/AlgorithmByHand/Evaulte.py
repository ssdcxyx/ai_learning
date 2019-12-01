# -*- coding: utf-8 -*-
# @time       : 4/11/2019 9:04 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @file       : Evaulte.py
# @description:
from time import time


# 运行时间
def run_time(fn):
    def fun():
        start = time()
        fn()
        ret = time() - start
        ret *= 1e3
        print("Total run time is %.1f ms\n" % ret)
    return fun()


# 准确率
def get_accuracy(y, y_pred):
    return sum(yi == yi_pred for yi, yi_pred in zip(y, y_pred)) / len(y)


# 精确率
def get_precision(y, y_pred):
    true_positive = sum(yi and yi_pred for yi, yi_pred in zip(y, y_pred))
    predicted_positive = sum(y_pred)
    return true_positive / predicted_positive


# 召回率
def get_recall(y, y_pred):
    true_positive = sum(yi and yi_pred for yi, yi_pred in zip(y, y_pred))
    actual_positive = sum(y)
    return true_positive / actual_positive


def get_r2(y, y_pred):
    sse = sum((yi - yi_pred) ** 2 for yi, yi_pred in zip(y, y_pred))
    y_avg = sum(y) / len(y)
    sst = sum((yi - y_avg) ** 2 for yi in y)
    r2 = 1 - sse / sst
    return r2


# 真正率
def get_tpr(y, y_pred):
    return get_recall(y, y_pred)


# 真负率
def get_tnr(y, y_pred):
    true_negative = sum(1 - (yi or yi_pred) for yi, yi_pred in zip(y, y_pred))
    actual_negative = len(y) - sum(y)
    return true_negative / actual_negative


# 假正率
def get_fpr(y, y_pred):
    return 1-get_tnr(y, y_pred)


# 假负率
def get_fnr(y, y_pred):
    return 1-get_tpr(y, y_pred)


# 受试者工作曲线
def get_roc(y, y_pred_prob):
    thresholds = sorted(set(y_pred_prob), reverse=True)
    ret = [[0, 0]]
    for threshold in thresholds:
        y_hat = [int(yi_pred_prob >= threshold) for yi_pred_prob in y_pred_prob]
        ret.append([get_tpr(y, y_hat), get_fpr(y, y_hat)])
    return ret


# Roc曲线下的面积
def get_auc(y, y_pred_prob):
    roc = iter(get_roc(y, y_pred_prob))
    tpr_pre, fpr_pre = next(roc)
    auc = 0
    for tpr, fpr in roc:
        auc += (tpr + tpr_pre) * (fpr - fpr_pre) / 2
        tpr_pre = tpr
        fpr_pre = fpr
    return auc

