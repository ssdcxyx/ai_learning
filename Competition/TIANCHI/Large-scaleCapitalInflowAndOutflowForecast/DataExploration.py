# -*- coding: utf-8 -*-
# @Time    : 2019-01-28 10:29
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : DataExploration.py

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.seasonal import seasonal_decompose
import statsmodels.api as sm


import os
from datetime import datetime

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.dates as mdate
from matplotlib.ticker import Formatter
plt.rcParams['axes.labelsize'] = 14
plt.rcParams['xtick.labelsize'] = 12
plt.rcParams['ytick.labelsize'] = 12

from Item import Item

PROJECT_ROOT_DIR = "."


# 保存图片
def save_fig(fig_id, tight_layout=True):
    path = os.path.join(PROJECT_ROOT_DIR, "images", fig_id + ".png")
    print("Saving figure", fig_id)
    if tight_layout:
        plt.tight_layout()
    plt.savefig(path, format="png", dpi=300)


# 原始数据位置
original_data_path = "img/original/"
# 处理完成之后的数据位置
handle_data_path = 'img/handle/'

# 原始用户信息表
user_profile_table = original_data_path + 'user_profile_table.csv'
# 原始用户申购赎回数据表
user_balance_table = original_data_path + 'user_balance_table.csv'
# 原始收益率表
mfd_day_share_interest = original_data_path + 'mfd_day_share_interest.csv'
# 原始上海银行间同业拆放利率表
mfd_bank_shibor = original_data_path + "mfd_bank_shibor.csv"
# 原始结果表
comp_predict_table = original_data_path + 'comp_predict_table.csv'

# 处理过的数据表
# 去除空数据并按时间排序之后的表
user_balance_no_empty_table = handle_data_path + 'user_balance_no_empty_table.csv'
# 原始统计数据
all_user_balance_daily_table = handle_data_path + 'all_user_balance_daily.csv'
# 统计每天详细数据并进行数据预处理之后的表
all_user_balance_daily_handled_table = handle_data_path + 'all_user_balance_daily_handled.csv'
# stl分解后的数据
decomposed_all_user_balance_daily_table = handle_data_path + 'decomposed_all_user_balance_daily_table.csv'


# 获得原始数据
def get_original_input_data():
    # 原始用户信息数据
    user_profile_o_data = pd.read_csv(user_profile_table, encoding="GBK")
    # 原始用户申购赎回数据
    user_balance_o_data = pd.read_csv(user_balance_table)
    # 收益率数据
    mfd_day_share_interest_o_data = pd.read_csv(mfd_day_share_interest)
    # 上海银行间同业拆放利率数据
    mfd_bank_shibor_o_data = pd.read_csv(mfd_bank_shibor)

    return user_profile_o_data, user_balance_o_data, mfd_day_share_interest_o_data, mfd_bank_shibor_o_data


# 去除全部数据为0即空数据或无效数据并排序
def wipe_empty_data(user_balance_o_data):
    user_balance_no_empty_data = user_balance_o_data[(user_balance_o_data['total_purchase_amt'] != 0)
                                            | (user_balance_o_data['total_redeem_amt'] != 0)].sort_values("report_date")
    user_balance_no_empty_data.to_csv(user_balance_no_empty_table, index=False, encoding='utf-8')
    return user_balance_no_empty_data


# 统计每天的数据
def statistical_all_user_balance_by_date(user_balance_no_empty_data, dates, handle=False):
    # sums_purchase_daily 日申购量
    # sums_redeem_daily 日赎回量
    # sum_consume_daily 日消费总量
    # sum_transfer_daily 日转出总量
    # amounts_purchase_daily 日申购操作量
    # amounts_redeem_daily 日赎回操作量
    # amounts_consume_daily 日消费操作数
    # amounts_transfer_daily 日转出操作数
    items = [Item("sums_purchase_daily", []), Item('sums_redeem_daily', []), Item('sums_consume_daily', []),
             Item('sums_transfer_daily', []),
             Item("amounts_purchase_daily", []), Item('amounts_redeem_daily', []), Item('amounts_consume_daily', []),
             Item('amounts_transfer_daily', [])]
    date_index = 0
    data = [[0 for x in range(0, 2)] for y in range(0, len(items))]
    new_dates = [20130701]
    date = None
    for row in user_balance_no_empty_data.itertuples():
        date = getattr(row, 'report_date')
        if date != dates[date_index]:
            if handle:
                # 2月份数据受长假影响较大，故剔除2月份的数据
                if 20140201 <= date <= 20140228:
                    data = [[0 for x in range(0, 2)] for y in range(0, len(items))]
                    date_index += 1
                else:
                    if date in [20130801, 20130901, 20131101, 20140101, 20140201, 2014401, 20140601, 20140801]:
                        for i in range(0, len(items)):
                            data[i][0] = (data[i][0] + data[i][1]) // 2 + 1
                    if date in [20130731, 20130831, 20131031, 20131231, 20140131, 20140331, 20140531, 20140731,
                                20140831]:
                        for i in range(0, len(items)):
                            data[i][1] = data[i][0]
                            data[i][0] = 0
                        date_index += 1
                    else:
                        new_dates.append(date)
                        for i in range(0, len(items)):
                            items[i].value.append(data[i][0])
                        # 初始化
                        data = [[0 for x in range(0, 2)] for y in range(0, len(items))]
                        date_index += 1
            else:
                new_dates.append(date)
                for i in range(0, len(items)):
                    items[i].value.append(data[i][0])
                # 初始化
                data = [[0 for x in range(0, 2)] for y in range(0, len(items))]
                date_index += 1
        total_purchase_amt = getattr(row, 'total_purchase_amt')
        # 剔除申购数据为0的数据
        if total_purchase_amt > 0:
            data[0][0] += total_purchase_amt
            data[4][0] += 1
        total_redeem_amt = getattr(row, 'total_redeem_amt')
        # 剔除赎回数据为0的数据
        if total_redeem_amt > 0:
            data[1][0] += total_redeem_amt
            data[5][0] += 1
        consume_amt = getattr(row, 'consume_amt')
        # 剔除消费数据为0的数据
        if consume_amt != 0:
            data[2][0] += consume_amt
            data[6][0] += 1
        transfer_amt = getattr(row, 'transfer_amt')
        # 剔除赎回数据为0的数据
        if transfer_amt != 0:
            data[3][0] += transfer_amt
            data[7][0] += 1
    # 最后一天
    if handle:
        for i in range(0, len(items)):
            data[i][0] = (data[i][0] + data[i][1]) // 2
        for i in range(0, len(items)):
            items[i].value.append(data[i][0])
    else:
        for i in range(0, len(items)):
            items[i].value.append(data[i][0])
    dates = [datetime.strptime(str(d), '%Y%m%d').date() for d in new_dates]
    all_user_balance_daily = pd.DataFrame({'report_date': dates,
                                           'sums_purchase_daily': items[0].value,
                                           'sums_redeem_daily': items[1].value,
                                           'sums_consume_daily': items[2].value,
                                           'sums_transfer_daily': items[3].value,
                                           'amounts_purchase_daily': items[4].value,
                                           'amounts_redeem_daily': items[5].value,
                                           'amounts_consume_daily': items[6].value,
                                           'amounts_transfer_daily': items[7].value},
                                          index=pd.to_datetime(dates),
                                          columns=['report_date', items[0].name, items[1].name, items[2].name,
                                                   items[3].name, items[4].name, items[5].name, items[6].name,
                                                   items[7].name])
    if handle:
        all_user_balance_daily.to_csv(all_user_balance_daily_handled_table, index=False, encoding='utf-8')
    else:
        all_user_balance_daily.to_csv(all_user_balance_daily_table, index=False, encoding='utf-8')
    return all_user_balance_daily


def month_decompose_pretreatment(timeseries):
    # 月周期分解只使用4、5、6、7、8月的数据
    timeseries = timeseries.truncate(before='2014-03-01')
    # 月天数对齐
    for date in ['2014-03-30', '2014-05-30', '2014-07-30', '2014-08-30']:
        day = datetime.strptime(date, '%Y-%m-%d')
        next_day = day + pd.tseries.offsets.DateOffset(days=1)
        day = datetime.strftime(day, '%Y-%m-%d')
        next_day = datetime.strftime(next_day, '%Y-%m-%d')
        next_day_data = pd.Series(np.array(timeseries[next_day:next_day]).tolist(),
                                  index=timeseries[day:day].index)
        timeseries[day] = (timeseries[day] + next_day_data)/2
        timeseries = timeseries[~timeseries.index.isin([next_day])]
    return timeseries


def week_decompose_pretreatment(timeseries):
    # 剔除2月份的数据
    timeseries1 = timeseries.truncate(after='2014-01-31')
    timeseries2 = timeseries.truncate(before='2014-03-01')
    return timeseries1.append(timeseries2)


# 检测数据的平稳性(adf: Augmented Dickey-Fuller Test)
def test_stationarity(timeseries, handle_name=''):
    # 去除空值
    timeseries = timeseries.dropna(how=any)
    # Dickey-Fuller test:
    print('Results of Dickey-Fuller Test:(' + handle_name + ')')
    dftest = adfuller(timeseries, autolag='AIC')
    # dftest的输出前一项依次为统计量，p值，滞后数，使用的观测数，各个置信度下的临界值
    dfoutput = pd.Series(dftest[0:4], index=['Test Statistic', 'p-value', '#Lags Used', 'Number of Observations Used'])
    for key, value in dftest[4].items():
        dfoutput['Critical value (%s)' % key] = value
    # 当statistic < critical value时认为在一定置信水平下数据是稳定的，反之是非稳定的。
    print(dfoutput)
    return


# 移动平均法
def moving_average(timeseries, log=True, ew=True):
    if log is True:
        timeseries = np.log(timeseries)
    if ew is True:
        moving_avg = timeseries.ewm(halflife=30, min_periods=0, adjust=True, ignore_na=False).mean()
    else:
        moving_avg = timeseries.rolling(window=30, center=False).mean()
    plt.plot(timeseries, color='blue')
    plt.plot(moving_avg, color='red')
    ts_log_moving_avg_diff = timeseries - moving_avg
    ts_log_moving_avg_diff.dropna(inplace=True)
    test_stationarity(ts_log_moving_avg_diff)
    return moving_avg


# STL分解
def decompose(timeseries, freq, model="multiplicative"):
    # 去除空值
    timeseries = timeseries.dropna(how=any)
    decomposition = seasonal_decompose(timeseries, model=model, freq=freq, two_sided=False)
    # decomposition.plot()
    trend = decomposition.trend
    seasonal = decomposition.seasonal
    residual = decomposition.resid
    return trend, seasonal, residual


# 差分
def difference(data, column=''):
    # 一阶差分
    diff1_data = data.diff(1)
    # 二阶差分
    diff2_data = diff1_data.diff(1)
    plot_data_comparison(data, [diff1_data, diff2_data], column=column,
                         titles=['log', 'diff(1)', 'diff(2)'])
    # 测试平稳性
    test_stationarity(data, handle_name='original')
    test_stationarity(diff1_data, handle_name='diff(1)')
    test_stationarity(diff2_data, handle_name='diff(2)')


class MyFormatter(Formatter):
    def __init__(self, dates, fmt='%Y-%m'):
        self.dates = dates
        self.fmt = fmt

    def __call__(self, x, pos=0):
        ind = int(np.round(x))
        if ind >= len(self.dates) or ind < 0:
            return ''

        a = self.dates[ind].strftime(self.fmt)
        return a


# 显示数据
def plot_data(data, display_columns, title=''):
    colors = ['black', 'gray', 'darkgrey']
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(111)
    ax.xaxis.set_major_formatter(MyFormatter(data[0].index))
    for i in range(0, len(data)):
        plt.plot(np.arange(len(data[i])), data[i], "-", label=display_columns[i], color=colors[i])
    plt.xlabel("date")
    plt.ylabel("amount")
    plt.title(title)
    if len(data) > 1:
        plt.legend()
    fig.autofmt_xdate()


# 数据对比
def plot_data_comparison(data, handled_data, column, titles, is_original, xtick=True):
    fig = plt.figure(figsize=(20, 14))
    ax = fig.add_subplot(100 + (len(handled_data) + 1) * 10 + 1)
    ax.xaxis.set_major_formatter(MyFormatter(data.index))
    plt.plot(np.arange(len(data)), data,
             "-", label=column, color="black")
    plt.title(titles[0])
    plt.legend()
    plt.xlabel("date")
    plt.ylabel("amount")
    if xtick:
        plt.xticks(np.arange(0, 450, 30), ['', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12',
                                           '2014-01',
                                           '2014-02', '2014-03', '2014-04', '2014-05', '2014-06', '2014-07',
                                           '2014-08'])
    for i in range(0, len(handled_data)):
        ax = fig.add_subplot(100 + (len(handled_data) + 1) * 10 + 1 + i + 1)
        ax.xaxis.set_major_formatter(MyFormatter(handled_data[i].index))
        plt.plot(np.arange(len(handled_data[i])), handled_data[i],
                 "-", label=column, color="black")
        plt.title(titles[i+1])
        plt.legend()
        plt.xlabel("date")
        plt.ylabel("amount")
        if xtick and is_original[i]:
            plt.xticks(np.arange(0, 450, 30), ['', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12',
                                               '2014-01',
                                               '2014-02', '2014-03', '2014-04', '2014-05', '2014-06', '2014-07',
                                               '2014-08'])
        else:
            plt.xticks(np.arange(0, 420, 30), ['', '2013-07', '2013-08', '2013-09', '2013-10', '2013-11', '2013-12',
                                               '2014-01', '2014-03', '2014-04', '2014-05', '2014-06', '2014-07',
                                               '2014-08'])
    fig.autofmt_xdate()


# ACF图和PACF图
def plot_acf_pacf_pic(data, lags):
    data = data.dropna(how=any)
    fig = plt.figure(figsize=(20, 14))
    ax1 = fig.add_subplot(211)
    fig = sm.graphics.tsa.plot_acf(data, lags=lags, ax=ax1)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout()
    ax2 = fig.add_subplot(212)
    fig = sm.graphics.tsa.plot_pacf(data, lags=lags, ax=ax2)
    ax1.xaxis.set_ticks_position('bottom')
    fig.tight_layout()


def plot_rolling_comparison(timeseries):
    fig = plt.figure(figsize=(20, 14))
    # 加权移动平均
    rolmean_weights = [[] for i in range(0, len(timeseries))]
    rolstds = [[] for i in range(0, len(timeseries))]
    for i in range(0, len(timeseries)):
        timeseries[i] = timeseries[i].dropna(how=any)
        rolmean_weights[i] = timeseries[i].rolling(window=90).mean()
        # rolmean_weights[i] = timeseries[i].ewm(span=90).mean()
        ax = fig.add_subplot(100 + len(timeseries) * 10 + i + 1)
        ax.xaxis.set_major_formatter(MyFormatter(timeseries[i].index))
        plt.plot(np.arange(len(timeseries[i])), timeseries[i], "-", label='original', color='black')
        plt.plot(np.arange(len(rolmean_weights[i])), rolmean_weights[i], "-", label='rolling mean weight', color='gray')
        plt.legend()
        plt.xlabel("date")
        plt.ylabel("amount")


def balance_data_distribution(timeseries):
    items = [Item('total_purchase_amt', 0), Item('direct_purchase_amt', 0), Item('purchase_bal_amt', 0),
             Item('purchase_bank_amt', 0),
             Item('total_redeem_amt', 0), Item('consume_amt', 0), Item('transfer_amt', 0),
             Item('tftocard_amt', 0), Item('share_amt', 0), Item('category1', 0), Item('category2', 0),
             Item('category3', 0), Item('category4', 0)]
    # items = [Item('category1', 0), Item('category2', 0),
    #          Item('category3', 0), Item('category4', 0)]
    for row in timeseries.itertuples():
        for i in range(0, len(items)):
            data = getattr(row, items[i].name)
            if pd.isnull(data):
                items[i].value += 1
            # if img == 0:
            #     items[i].value += 1
    x = [i for i in range(0, len(items))]
    x_label = [item.name for item in items]
    y = [(item.value/len(timeseries)) for item in items]
    plt.xticks(x, x_label, rotation=15)
    plt.bar(x, y)


def get_handle_data(data_column):
    # 获取原始数据
    # user_profile_o_data, user_balance_o_data, mfd_day_share_interest_o_data, mfd_bank_shibor_o_data =\
    #     get_original_input_data()
    # 交易数据分布
    # balance_data_distribution(user_balance_o_data)
    # 丢弃申购赎回为0的数据并排序
    # user_balance_no_empty_data = wipe_empty_data(user_balance_o_data)

    # 统计每天的申购和赎回数据
    # user_balance_no_empty_data = pd.read_csv(user_balance_no_empty_table)
    # train_dates = pd.unique(user_balance_no_empty_data.report_date)
    # all_user_balance_daily = statistical_all_user_balance_by_date(user_balance_no_empty_data, train_dates, handle=True)

    # 显示每天的数据
    dateparse = lambda dates: pd.datetime.strptime(dates, '%Y-%m-%d')
    all_user_balance_daily = pd.read_csv(all_user_balance_daily_table, index_col='report_date',
                                         parse_dates=['report_date'], date_parser=dateparse, skipinitialspace=True)
    all_user_balance_daily_handled = pd.read_csv(all_user_balance_daily_handled_table, index_col='report_date',
                                                 parse_dates=['report_date'], date_parser=dateparse,
                                                 skipinitialspace=True)
    # sums_purchase_daily 日申购量
    # sums_redeem_daily 日赎回量
    # sum_consume_daily 日消费总量
    # sum_transfer_daily 日转出总量
    # amounts_purchase_daily 日申购操作量
    # amounts_redeem_daily 日赎回操作量
    # amounts_consume_daily 日消费操作数
    # amounts_transfer_daily 日转出操作数
    sums_balance_daily_columns = ['sums_purchase_daily', 'sums_redeem_daily',
                                  'sums_consume_daily', 'sums_transfer_daily']
    amounts_balance_daily_columns = ['amounts_purchase_daily', 'amounts_redeem_daily',
                                     'amounts_consume_daily', 'amounts_transfer_daily']
    # 数据显示与对比
    # plot_data([all_user_balance_daily['sums_transfer_daily']],
    #           display_columns='sums_transfer_daily',
    #           title='sums_transfer_daily')
    # data1 = all_user_balance_daily['sums_consume_daily'] + all_user_balance_daily['sums_transfer_daily']
    # plot_data_comparison(all_user_balance_daily['sums_transfer_daily'],
    #                      [all_user_balance_daily_handled['sums_transfer_daily']], column='sums_transfer_daily',
    #                      titles=['original', 'real'], is_original=[False])
    # plot_rolling_comparison([all_user_balance_daily_handled['sums_redeem_daily'],
    #                          np.log(all_user_balance_daily_handled['sums_redeem_daily'])])
    # plot_data([all_user_balance_daily_handled.sums_purchase_daily,
    #            all_user_balance_daily_handled.sums_redeem_daily,
    #            all_user_balance_daily_handled.sums_purchase_redeem_daily],
    #           display_columns=sums_balance_daily_columns,
    #           title='all_user_balance_daily_handle')
    # stl 分解
    # 星期周期分解
    week_decompose_pre_data = week_decompose_pretreatment(all_user_balance_daily[data_column])
    week_decompose_pre_data = week_decompose_pre_data.dropna(how=any)
    week_trend, week_seasonal, week_residual = decompose(week_decompose_pre_data, freq=7)
    # 月周期分解
    month_decompose_pre_data = month_decompose_pretreatment(week_residual * week_trend)
    month_trend, month_seasonal, month_residual = decompose(month_decompose_pre_data, freq=30)
    # 季度分解
    # month_seasonal = month_trend.dropna(how=any)
    # season_trend, season_seasonal, season_residual = decompose(month_trend, freq=90)
    # month_trend = season_trend
    # plot_rolling_comparison([month_trend.diff(1).diff(1)])
    # test_stationarity(month_trend.diff(1).diff(1))
    # 移动平均
    # moving_average(all_user_balance_daily['sums_purchase_redeem_daily'], log=False, ew=True)
    # Difference 差分
    # test_stationarity(all_user_balance_daily_handled['sums_purchase_daily'].diff(7))
    # plot_acf_pacf_pic(np.log(all_user_balance_daily_handled['sums_purchase_daily']).diff(1))
    # Decomposing 分解
    # decompose(all_user_balance_daily, log=True, date=None, freq=30)
    month_trend = month_trend.dropna(how=any)
    month_residual = month_residual.dropna(how=any)
    # test_stationarity(month_residual)
    # plot_acf_pacf_pic(month_residual, lags=149)
    # plt.show()
    return {'residual': month_residual, 'trend': month_trend, 'month_seasonal': month_seasonal,
            'week_seasonal': week_seasonal}


if __name__ == "__main__":
    get_handle_data()


