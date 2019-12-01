# -*- coding: utf-8 -*-
# @Time    : 2019-05-05 16:33
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Main.py

from DataExploration import get_handle_data
from BuildModel import build_arima_model
from BuildModel import build_svm_model
import BuildModel
import DataExploration
import os
from Feature import Feature

import pandas as pd
import time
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# 原始数据位置
result_data_path = "img/result/"

ARIMA_result_table = result_data_path + 'arima.csv'
result_table = result_data_path + 'result.csv'


if __name__ == "__main__":
    columns = [Feature('sums_purchase_daily', 0, 0, 1),
               Feature('sums_consume_daily', 1, 0, 0),
               Feature('sums_transfer_daily', 1, 0, 1)]
    categories = ['residual', 'trend']
    # 预测目标表
    forest_columns = ['purchase', 'consume', 'transfer']
    forest_data = [[], [], []]
    forest_total_redeem = []
    forest_date = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start='2014-09-01', end='2014-09-30'))]
    for i in range(0, len(columns)):
        data = get_handle_data(columns[i].name)
        for category in categories:
            # 参数估计
            # BuildModel.parameter_estimation(img['trend'])
            if category == 'residual':
                # 余数预测
                forest_r = build_arima_model(data[category], columns[i].p, columns[i].d, columns[i].q,
                                             predict=False, show_qqplot=False)
            else:
                # 趋势预测
                if columns[i].name == 'sums_transfer_daily':
                    # BuildModel.parameter_estimation(img['trend'])
                    forest_t = build_arima_model(data[category], predict=True, show_qqplot=False)
                else:
                    forest_t = build_svm_model(data[category])
            # 月周期变动
            forest_m_s = data['month_seasonal']['2014-04-01':'2014-04-30']
            # 星期周期变动
            forest_w_s = data['week_seasonal']['2014-04-07':'2014-05-06']
        for j in range(0, 30):
            forest_data[i].append(int(forest_r['forest'].values[j] * forest_t['forest'].values[j]
                                      * forest_m_s[j] * forest_w_s[j]))
    for i in range(0, 30):
        forest_total_redeem.append(forest_data[1][i] + forest_data[2][i])
    forest_df = pd.DataFrame({'sums_purchase_daily': forest_data[0], 'sums_redeem_daily': forest_total_redeem},
                             index=pd.to_datetime(forest_date), columns=['sums_purchase_daily', 'sums_redeem_daily'])
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    all_user_balance_daily = pd.read_csv(DataExploration.all_user_balance_daily_table, index_col='report_date',
                                         parse_dates=['report_date'], date_parser=dateparse, skipinitialspace=True)
    past_purchase_df = all_user_balance_daily['sums_purchase_daily']
    fig1, ax1 = plt.subplots(figsize=(12, 8))
    ax1 = past_purchase_df.plot(ax=ax1)
    ax1 = forest_df['sums_purchase_daily'].plot(ax=ax1)
    past_redeem_df = all_user_balance_daily['sums_redeem_daily']
    fig2, ax2 = plt.subplots(figsize=(12, 8))
    ax2 = past_redeem_df.plot(ax=ax2)
    ax2 = forest_df['sums_redeem_daily'].plot(ax=ax2)
    csv_date = forest_date = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start='2014-09-01', end='2014-09-30'))]
    forest_df.to_csv(result_table, date_format='%Y%m%d', index=csv_date, encoding='utf-8')
    plt.show()



