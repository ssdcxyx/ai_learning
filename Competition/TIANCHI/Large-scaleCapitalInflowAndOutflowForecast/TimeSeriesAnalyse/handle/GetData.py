# -*- coding: utf-8 -*-
# @Time    : 2019-05-10 10:17
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : GetData.py


import pandas as pd

import os

from setting import Constant
from handle import DataProcess
from handle import File


def get_data(file_name, encoding='utf-8', parse_dates=None, date_parser=None):
    if parse_dates is None:
        return pd.read_csv(file_name, encoding=encoding)
    else:
        return pd.read_csv(file_name, encoding=encoding, parse_dates=parse_dates, date_parser=date_parser)


# 获得用户分类训练数据
def get_user_classification_data():
    # user_profile = get_data(file_name=Constant.user_profile_table, encoding="GBK").sort_values(
    #     by="user_id", ascending=True)
    user_max_balance = get_data(file_name=Constant.user_max_balance_table)
    user_max_consume = get_data(file_name=Constant.user_max_consume_table)
    user_max_purchase = get_data(file_name=Constant.user_max_purchase_table)
    user_max_transfer = get_data(file_name=Constant.user_max_transfer_table)
    user_average_purchase = get_data(file_name=Constant.user_average_purchase_table)
    user_average_transfer = get_data(file_name=Constant.user_average_transfer_table)
    user_average_category1 = get_data(file_name=Constant.user_average_category1_table)
    user_average_category2 = get_data(file_name=Constant.user_average_category2_table)
    user_average_category3 = get_data(file_name=Constant.user_average_category3_table)
    user_average_category4 = get_data(file_name=Constant.user_average_category4_table)
    user_monthly_consume_count = get_data(file_name=Constant.user_monthly_consume_count_table)
    user_monthly_purchase_count = get_data(file_name=Constant.user_monthly_purchase_count_table)
    user_monthly_transfer_count = get_data(file_name=Constant.user_monthly_transfer_count_table)
    user_profile_handle = get_data(file_name=Constant.user_profile_handle_table)
    user_classification_data = pd.concat([user_profile_handle,
                                          user_max_balance['max_balance'],
                                          user_max_consume['max_consume'],
                                          user_max_purchase['max_purchase'],
                                          user_max_transfer['max_transfer'],
                                          user_average_purchase['average_purchase'],
                                          user_average_transfer['average_transfer'],
                                          user_average_category1['average_category1'],
                                          user_average_category2['average_category2'],
                                          user_average_category3['average_category3'],
                                          user_average_category4['average_category4'],
                                          user_monthly_consume_count['monthly_consume_count'],
                                          user_monthly_purchase_count['monthly_purchase_count'],
                                          user_monthly_transfer_count['monthly_transfer_count']], axis=1)
    File.store_csv(user_classification_data, file_path=Constant.user_classification_data_table)
    return user_classification_data


if __name__ == "__main__":
    # 获得用户分类训练数据
    get_user_classification_data()
    print()

