# -*- coding: utf-8 -*-
# @Time    : 2019-05-10 10:19
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Constant.py


"""
文件地址
"""
# 根目录
PROJECT_ROOT_DIR = ".."
DATA_FILE_PATH = "/img"

# 原始文件
original_data_path = PROJECT_ROOT_DIR + DATA_FILE_PATH + "/original/"
# 用户信息
user_profile_table = original_data_path + 'user_profile_table.csv'
# 用户申购赎回数据
user_balance_table = original_data_path + 'user_balance_table.csv'
# 收益率表
mfd_day_share_interest_table = original_data_path + 'mfd_day_share_interest.csv'
# 上海银行间同业拆放利率
mfd_bank_shibor_table = original_data_path + "mfd_bank_shibor.csv"
# 结果
comp_predict_table = original_data_path + 'comp_predict_table.csv'

# 预处理文件
data_preprocessing_path = PROJECT_ROOT_DIR + DATA_FILE_PATH + "/real/"
# 缺失值填充并剔除全为0的数据
fill_missing_value_wipe_empty_balance_table = data_preprocessing_path +\
                                              "fill_missing_value_wipe_empty_balance_table.csv"
# 用户信息处理数据
user_profile_handle_table = data_preprocessing_path + "user_profile_handle.csv"
# 用户分类的训练数据
user_classification_data_table = data_preprocessing_path + "user_classification_data_table.csv"


# 提取的特征存储文件
feature_path = PROJECT_ROOT_DIR + DATA_FILE_PATH + "/feature/"
# 用户注册日期
user_register_date_table = feature_path + "user_register_date.csv"
# 用户增长情况
user_register_growth_table = feature_path + 'user_register_growth.csv'
# 用户第一次交易日期
user_first_trading_date_table = feature_path + "user_first_trading_date.csv"
# 用户第一次交易增长情况
user_first_trading_growth_table = feature_path + "user_first_trading_growth.csv"
# 用户最大余额
user_max_balance_table = feature_path + "user_max_balance.csv"
# 用户最大消费
user_max_consume_table = feature_path + "user_max_consume.csv"
# 用户最大申购
user_max_purchase_table = feature_path + "user_max_purchase.csv"
# 用户最大转出
user_max_transfer_table = feature_path + "user_max_transfer.csv"

# 用户平均申购额
user_average_purchase_table = feature_path + "user_average_purchase.csv"
# 用户平均转出额
user_average_transfer_table = feature_path + "user_average_transfer.csv"

# 用户类别1消费平均值
user_average_category1_table = feature_path + "user_average_category1.csv"
# 用户类别2消费平均值
user_average_category2_table = feature_path + "user_average_category2.csv"
# 用户类别3消费平均值
user_average_category3_table = feature_path + "user_average_category3.csv"
# 用户类别4消费平均值
user_average_category4_table = feature_path + "user_average_category4.csv"

# 用户月均申购操作数
user_monthly_purchase_count_table = feature_path + "user_monthly_purchase_count.csv"
# 用户月均消费操作数
user_monthly_consume_count_table = feature_path + "user_monthly_consume_count.csv"
# 用户月均转出操作数
user_monthly_transfer_count_table = feature_path + "user_monthly_transfer_count.csv"

# 结果文件
result_path = PROJECT_ROOT_DIR + DATA_FILE_PATH + "/result/"
# 用户分类结果
user_classification_result_table = result_path + "user_classification_result.csv"


