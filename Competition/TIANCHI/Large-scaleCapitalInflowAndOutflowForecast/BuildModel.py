# -*- coding: utf-8 -*-
# @Time    : 2019-05-05 16:33
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : BuildModel.py

import matplotlib
import matplotlib.pyplot as plt
from pyramid.arima import auto_arima
from geneview.gwas import qqplot
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from scipy import stats
import statsmodels.api as sm
import numpy as np
import pandas as pd
import itertools
import seaborn as sns


# 参数估计
def parameter_estimation(data):
    p_min, p_max = 0, 5
    d_min, d_max = 2, 2
    q_min, q_max = 0, 5
    p_max = 5
    results_aic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                               columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
    results_bic = pd.DataFrame(index=['AR{}'.format(i) for i in range(p_min, p_max + 1)],
                               columns=['MA{}'.format(i) for i in range(q_min, q_max + 1)])
    for p, d, q in itertools.product(range(p_min, p_max + 1),
                                     range(d_min, d_max + 1),
                                     range(q_min, q_max + 1)):
        if p == 0 and d == 1 and q == 0:
            results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = np.nan
            continue
        try:
            model = sm.tsa.ARIMA(data, order=(p, d, q))
            results = model.fit()
            results_aic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.aic
            results_bic.loc['AR{}'.format(p), 'MA{}'.format(q)] = results.bic
        except:
            continue
    results_aic = results_aic[results_aic.columns].astype(float)
    results_bic = results_bic[results_bic.columns].astype(float)
    fig1, ax1 = plt.subplots(figsize=(10, 8))
    ax1 = sns.heatmap(results_aic, mask=results_aic.isnull(), ax=ax1, annot=True, fmt='.2f')
    ax1.set_title('AIC')
    fig2, ax2 = plt.subplots(figsize=(10, 8))
    ax2 = sns.heatmap(results_bic, mask=results_bic.isnull(), ax=ax2, annot=True, fmt='.2f', )
    ax2.set_title('BIC')
    plt.show()


def build_arima_model(data, p=0, d=0, q=0, predict=False, show_qqplot=False):
    forest_date = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start='2014-09-01', end='2014-09-30'))]
    if p == 0 and d == 0 and q == 0:
        model = auto_arima(data)
        model.fit(data)
    else:
        model = sm.tsa.ARIMA(data, order=(p, d, q))
        results = model.fit()
        # 滚动预测
        # forest = []
        # for i in range(0, len(forest_date)-7):
        #     if i % 7 == 0:
        #         temp = []
        #         temp += results.forecast(steps=7)[0].tolist()
        #         temp_pd = pd.Series(temp, index=pd.to_datetime(forest_date[i:i + 7]))
        #         img = img.append(temp_pd)
        #         forest += temp
        #         model = None
        #         model = sm.tsa.ARIMA(img, order=(p, d, q))
        #         results = model.fit()
        # forest += results.forecast(steps=2)[0].tolist()
    if show_qqplot:
        resid = results.resid.values
        for i in range(0, len(resid)):
            resid[i] = round(resid[i], 8)
        qqplot(resid)
    # 模型历史数据预测
    if predict:
        if p == 0 and d == 0 and q == 0:
            predict = model.predict_in_sample(start=0, end=151, dynamic=False)
        else:
            predict = results.predict(start=str('2014-04-03'), end=str('2014-08-30'), dynamic=False)
        predict_date = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start='2014-04-01', end='2014-08-30'))]
        predict_df = pd.DataFrame({'forest': predict},
                                  index=pd.to_datetime(predict_date))
        fig, ax = plt.subplots(figsize=(12, 8))
        ax = data.plot(ax=ax)
        ax = predict_df.plot(ax=ax)
    if p == 0 and d == 0 and q == 0:
        forest = model.predict(n_periods=30)
    else:
        forest = results.forecast(steps=30)[0]
    forest_df = pd.DataFrame({'forest': forest},
                             index=pd.to_datetime(forest_date))
    fig, ax = plt.subplots(figsize=(12, 8))
    ax = data.plot(ax=ax)
    ax = forest_df.plot(ax=ax)
    plt.show()
    return forest_df


def build_svm_model(data):
    X = [[i] for i in range(0, len(data))]
    y = data.values.tolist()
    forest_X = [[i] for i in range(0, len(data) + 30)]
    poly = PolynomialFeatures(degree=2, include_bias=False)
    poly_X = poly.fit_transform(X)
    poly_forest_X = poly.fit_transform(forest_X)
    svm_poly_reg = LinearRegression()
    svm_poly_reg.fit(poly_X, y)
    forest = svm_poly_reg.predict(poly_forest_X)
    forest_date = [x.strftime('%Y-%m-%d') for x in list(pd.date_range(start='2014-04-04', end='2014-09-30'))]
    fig, ax = plt.subplots(figsize=(12, 8))
    forest_df = pd.DataFrame({'forest': forest},
                             index=pd.to_datetime(forest_date))
    ax = data.plot(ax=ax)
    ax = forest_df.plot(ax=ax)
    plt.show()
    return forest_df






