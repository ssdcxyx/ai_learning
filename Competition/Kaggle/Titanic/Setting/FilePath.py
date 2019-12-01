# -*- coding: utf-8 -*-
# @Time    : 2019-06-29 12:18
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : FilePath.py


"""
文件存储地址
"""
# 根目录
PROJECT_ROOT_DIR = ".."
DATA_FILE_PATH = "/img"

# 原始数据文件
original_data_path = PROJECT_ROOT_DIR + DATA_FILE_PATH + "/original/"
# 原始训练数据
original_train_data = original_data_path + 'train.csv'
# 原始测试数据
original_test_data = original_data_path + 'test.csv'

# 处理之后的数据文件
handled_data_path = PROJECT_ROOT_DIR + DATA_FILE_PATH + "/real/"
# 处理之后的训练数据
handled_train_data = handled_data_path + "train.csv"
# 处理之后的测试数据
handled_test_data = handled_data_path + "test.csv"

# 预测结果
predict_result = PROJECT_ROOT_DIR + DATA_FILE_PATH + "/result/"
# logistics_regression
logistics_regression = predict_result + 'logistics_regression/'
# logistics回归预测结果(包含原始属性)
logistics_regression_result_feature = logistics_regression + 'result(feature).csv'
# logistics回归预测结果
logistics_regression_result = logistics_regression + 'result.csv'
# logistics回归预测错误的例子
logistics_regression_classification_bad_cases = logistics_regression + "classification_bad_cases.csv"

# logistics_regression_bagging
logistics_regression_bagging = predict_result + 'logistics_regression_bagging/'
# logistics_regression_bagging预测结果(包含原始属性)
logistics_regression_bagging_result_feature = logistics_regression_bagging + 'result(feature).csv'
# logistics_regression_bagging预测结果
logistics_regression_bagging_result = logistics_regression_bagging + 'result.csv'
# logistics_regression_bagging预测错误的例子
logistics_regression_bagging_classification_bad_cases = logistics_regression_bagging + "classification_bad_cases.csv"
