# -*- coding: utf-8 -*-
# @Time    : 2019-05-10 16:13
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : FeatureExtraction.py

import pandas as pd

from handle.GetData import get_data
from setting import Constant
from handle import DataProcess
from handle import File


# 用户最大余额
def user_max_balance(data):
    users = data['user_id'].unique()
    user_max_balance_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        max_balance = user_data['tBalance'].max()
        user_max_balance_df = user_max_balance_df.append(
            pd.DataFrame({'max_balance': max_balance}, index={user}))
    user_max_balance_df.index.name = 'user_id'
    return user_max_balance_df


# 用户最大消费
def user_max_consume(data):
    users = data['user_id'].unique()
    user_max_consume_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        max_consume = user_data['consume_amt'].max()
        user_max_consume_df = user_max_consume_df.append(
            pd.DataFrame({'max_consume': max_consume}, index={user}))
    user_max_consume_df.index.name = 'user_id'
    return user_max_consume_df


# 用户最大申购
def user_max_purchase(data):
    users = data['user_id'].unique()
    user_max_purchase_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        max_purchase = user_data['total_purchase_amt'].max()
        user_max_purchase_df = user_max_purchase_df.append(
            pd.DataFrame({'max_purchase': max_purchase}, index={user}))
    user_max_purchase_df.index.name = 'user_id'
    return user_max_purchase_df


# 用户最大转出
def user_max_transfer(data):
    users = data['user_id'].unique()
    user_max_transfer_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        max_transfer = user_data['transfer_amt'].max()
        user_max_transfer_df = user_max_transfer_df.append(
            pd.DataFrame({'max_transfer': max_transfer}, index={user}))
    user_max_transfer_df.index.name = 'user_id'
    return user_max_transfer_df


# 用户注册日期
def user_register_date(data):
    users = data['user_id'].unique()
    user_register_date_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        register_date = pd.to_datetime(str(user_data['report_date'].min()))
        user_register_date_df = user_register_date_df.append(
            pd.DataFrame({'user_register_date': register_date}, index={user}))
    user_register_date_df.index.name = 'user_id'
    return user_register_date_df


# 用户注册增长情况
def user_register_growth(data):
    dates = [x.strftime('%Y%m%d') for x in list(pd.date_range(start='20130701', end='20140801'))]
    user_register_growth_df = pd.DataFrame()
    for date in dates:
        data_day = data[data.user_register_date.isin([date])]
        if data_day is not None:
            count_user_register_growth = len(data_day)
        else:
            count_user_register_growth = 0
        user_register_growth_df = user_register_growth_df.append(
            pd.DataFrame({'count_user_register_growth': count_user_register_growth},
                         index={pd.to_datetime(date)}),)
    user_register_growth_df.index.name = 'report_date'
    return user_register_growth_df


# 用户第一次交易日期
def user_first_trading_date(data):
    users = data['user_id'].unique()
    user_first_trading_date_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        first_trading_date = pd.to_datetime(str(user_data['report_date'].min()))
        user_first_trading_date_df = user_first_trading_date_df.append(
            pd.DataFrame({'user_first_trading_date': first_trading_date}, index={user}))
    user_first_trading_date_df.index.name = 'user_id'
    return user_first_trading_date_df


# 用户第一次交易增长情况
def user_first_trading_growth(data):
    dates = [x.strftime('%Y%m%d') for x in list(pd.date_range(start='20130701', end='20140801'))]
    user_first_trading_growth_df = pd.DataFrame()
    for date in dates:
        data_day = data[data.user_first_trading_date.isin([date])]
        if data_day is not None:
            count_user_first_trading_growth = len(data_day)
        else:
            count_user_first_trading_growth = 0
        user_first_trading_growth_df = user_first_trading_growth_df.append(
            pd.DataFrame({'count_user_first_trading_growth': count_user_first_trading_growth},
                         index={pd.to_datetime(date)}),)
    user_first_trading_growth_df.index.name = 'report_date'
    return user_first_trading_growth_df


# 用户类别1的


# 用户月均申购操作数
def user_monthly_purchase_count(data):
    users = data['user_id'].unique()
    user_monthly_purchase_count_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        purchase_count = 0
        first_trading_date = pd.to_datetime(str(user_data['report_date'].min()))
        end_data = pd.to_datetime('20140831')
        days = (end_data - first_trading_date).days + 1
        for index, row in user_data.iterrows():
            if row['total_purchase_amt'] > 0:
                purchase_count += 1
        monthly_purchase_count = purchase_count / days * 30
        user_monthly_purchase_count_df = user_monthly_purchase_count_df.append(
            pd.DataFrame({'monthly_purchase_count': monthly_purchase_count},
                         index={user}))
    user_monthly_purchase_count_df.index.name = 'user_id'
    return user_monthly_purchase_count_df


# 用户月均消费操作数
def user_monthly_consume_count(data):
    users = data['user_id'].unique()
    user_monthly_consume_count_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        consume_count = 0
        first_trading_date = pd.to_datetime(str(user_data['report_date'].min()))
        end_data = pd.to_datetime('20140831')
        days = (end_data - first_trading_date).days + 1
        for index, row in user_data.iterrows():
            if row['consume_amt'] > 0:
                consume_count += 1
        monthly_consume_count = consume_count / days * 30
        user_monthly_consume_count_df = user_monthly_consume_count_df.append(
            pd.DataFrame({'monthly_consume_count': monthly_consume_count},
                         index={user}))
    user_monthly_consume_count_df.index.name = 'user_id'
    return user_monthly_consume_count_df


# 用户月均转出操作数
def user_monthly_transfer_count(data):
    users = data['user_id'].unique()
    user_monthly_transfer_count_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        transfer_count = 0
        first_trading_date = pd.to_datetime(str(user_data['report_date'].min()))
        end_data = pd.to_datetime('20140831')
        days = (end_data - first_trading_date).days + 1
        for index, row in user_data.iterrows():
            if row['transfer_amt'] > 0:
                transfer_count += 1
        monthly_transfer_count = transfer_count / days * 30
        user_monthly_transfer_count_df = user_monthly_transfer_count_df.append(
            pd.DataFrame({'monthly_transfer_count': monthly_transfer_count},
                         index={user}))
    user_monthly_transfer_count_df.index.name = 'user_id'
    return user_monthly_transfer_count_df


# 用户平均申购额
def user_average_purchase(data):
    users = data['user_id'].unique()
    user_average_purchase_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        purchase_sum = 0
        purchase_count = 0
        average_purchase = 0
        for index, row in user_data.iterrows():
            if row['total_purchase_amt'] > 0:
                purchase_sum += row['total_purchase_amt']
                purchase_count += 1
        if purchase_count > 0:
            average_purchase = purchase_sum / purchase_count
        user_average_purchase_df = user_average_purchase_df.append(
            pd.DataFrame({'average_purchase': average_purchase},
                         index={user}))
    user_average_purchase_df.index.name = 'user_id'
    return user_average_purchase_df


# 用户平均转出额
def user_average_transfer(data):
    users = data['user_id'].unique()
    user_average_transfer_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        transfer_sum = 0
        transfer_count = 0
        average_transfer = 0
        for index, row in user_data.iterrows():
            if row['transfer_amt'] > 0:
                transfer_sum += row['transfer_amt']
                transfer_count += 1
        if transfer_count > 0:
            average_transfer = transfer_sum / transfer_count
        user_average_transfer_df = user_average_transfer_df.append(
            pd.DataFrame({'average_transfer': average_transfer},
                         index={user}))
    user_average_transfer_df.index.name = 'user_id'
    return user_average_transfer_df


# 用户类别1平均消费额
def user_average_category1(data):
    users = data['user_id'].unique()
    user_average_category1_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        if user == 3:
            print()
        category1_sum = 0
        category1_count = 0
        average_category1 = 0
        for index, row in user_data.iterrows():
            if row['category1'] > 0:
                category1_sum += row['category1']
                category1_count += 1
        if category1_count > 0:
            average_category1 = category1_sum / category1_count
        user_average_category1_df = user_average_category1_df.append(
            pd.DataFrame({'average_category1': average_category1},
                         index={user}))
    user_average_category1_df.index.name = 'user_id'
    return user_average_category1_df


# 用户类别2平均消费额
def user_average_category2(data):
    users = data['user_id'].unique()
    user_average_category2_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        category2_sum = 0
        category2_count = 0
        average_category2 = 0
        for index, row in user_data.iterrows():
            if row['category2'] > 0:
                category2_sum += row['category2']
                category2_count += 1
        if category2_count > 0:
            average_category2 = category2_sum / category2_count
        user_average_category2_df = user_average_category2_df.append(
            pd.DataFrame({'average_category2': average_category2},
                         index={user}))
    user_average_category2_df.index.name = 'user_id'
    return user_average_category2_df


def user_average_category3(data):
    users = data['user_id'].unique()
    user_average_category3_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        category3_sum = 0
        category3_count = 0
        average_category3 = 0
        for index, row in user_data.iterrows():
            if row['category3'] > 0:
                category3_sum += row['category3']
                category3_count += 1
        if category3_count > 0:
            average_category3 = category3_sum / category3_count
        user_average_category3_df = user_average_category3_df.append(
            pd.DataFrame({'average_category3': average_category3},
                         index={user}))
    user_average_category3_df.index.name = 'user_id'
    return user_average_category3_df


def user_average_category4(data):
    users = data['user_id'].unique()
    user_average_category4_df = pd.DataFrame()
    for user in users:
        user_data = data[data['user_id'] == user]
        category4_sum = 0
        category4_count = 0
        average_category4 = 0
        for index, row in user_data.iterrows():
            if row['category4'] > 0:
                category4_sum += row['category4']
                category4_count += 1
        if category4_count > 0:
            average_category4 = category4_sum / category4_count
        user_average_category4_df = user_average_category4_df.append(
            pd.DataFrame({'average_category4': average_category4},
                         index={user}))
    user_average_category4_df.index.name = 'user_id'
    return user_average_category4_df


if __name__ == '__main__':
    # 用户第一次交易日期
    # user_balance_data_handled = get_data(file_name=Constant.fill_missing_value_wipe_empty_balance_table)
    # user_first_trading_date_data = user_first_trading_date(user_balance_data_handled)
    # File.store_csv(user_first_trading_date_data, file_path=Constant.user_first_trading_date_table, index=True)

    # 用户第一次交易增长情况
    # user_first_trading_date_data = get_data(file_name=Constant.user_first_trading_date_table)
    # user_first_trading_growth_data = user_first_trading_growth(user_first_trading_date_data)
    # File.store_csv(user_first_trading_growth_data, file_path=Constant.user_first_trading_growth_table, index=True)

    user_balance_data_handled = get_data(file_name=Constant.fill_missing_value_wipe_empty_balance_table)
    # 用户最大余额
    # user_max_balance_data = user_max_balance(user_balance_data_handled)
    # File.store_csv(user_max_balance_data, file_path=Constant.user_max_balance_table, index=True)
    # 用户最大消费
    # user_max_consume_data = user_max_consume(user_balance_data_handled)
    # File.store_csv(user_max_consume_data, file_path=Constant.user_max_consume_table, index=True)
    # 用户最大申购
    # user_max_purchase_data = user_max_purchase(user_balance_data_handled)
    # File.store_csv(user_max_purchase_data, file_path=Constant.user_max_purchase_table, index=True)
    # 用户最大转出
    # user_max_transfer_data = user_max_transfer(user_balance_data_handled)
    # File.store_csv(user_max_transfer_data, file_path=Constant.user_max_transfer_table, index=True)

    # 用户月均申购操作数
    # user_monthly_purchase_count_data = user_monthly_purchase_count(user_balance_data_handled)
    # File.store_csv(user_monthly_purchase_count_data, file_path=Constant.user_monthly_purchase_count_table, index=True)
    # 用户月均消费操作数
    # user_monthly_consume_count_data = user_monthly_consume_count(user_balance_data_handled)
    # File.store_csv(user_monthly_consume_count_data, file_path=Constant.user_monthly_consume_count_table, index=True)
    # 用户月均转出操作数
    # user_monthly_transfer_count = user_monthly_transfer_count(user_balance_data_handled)
    # File.store_csv(user_monthly_transfer_count, file_path=Constant.user_monthly_transfer_count_table, index=True)
    # 用户平均申购额
    # user_average_purchase_data = user_average_purchase(user_balance_data_handled)
    # File.store_csv(user_average_purchase_data, file_path=Constant.user_average_purchase_table, index=True)
    # 用户平均转出额
    user_average_transfer_data = user_average_transfer(user_balance_data_handled)
    File.store_csv(user_average_transfer_data, file_path=Constant.user_average_transfer_table, index=True)
    # 用户类别1平均消费
    # user_average_category1_data = user_average_category1(user_balance_data_handled)
    # File.store_csv(user_average_category1_data, file_path=Constant.user_average_category1_table, index=True)
    # 用户类别2平均消费
    # user_average_category2_data = user_average_category2(user_balance_data_handled)
    # File.store_csv(user_average_category2_data, file_path=Constant.user_average_category2_table, index=True)
    # 用户类别3平均消费
    # user_average_category3_data = user_average_category3(user_balance_data_handled)
    # File.store_csv(user_average_category3_data, file_path=Constant.user_average_category3_table, index=True)
    # 用户类别4平均消费
    # user_average_category4_data = user_average_category4(user_balance_data_handled)
    # File.store_csv(user_average_category4_data, file_path=Constant.user_average_category4_table, index=True)
    print()

