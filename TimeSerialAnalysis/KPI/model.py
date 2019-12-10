# -*- coding: utf-8 -*-
# @time       : 10/12/2019 8:30 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @description: 

from pyculiarity import detect_ts
from datetime import datetime
import matplotlib.pyplot as plt
import pandas as pd
from adtk.data import validate_series
from adtk.visualization import plot
from adtk.detector import *
from adtk.data import to_events
from adtk.pipe import Pipeline
from adtk.transformer import *
from luminol.anomaly_detector import *

from TimeSerialAnalysis.KPI.tool import get_data
from TimeSerialAnalysis.KPI.config import real_data_path, img_path


def twitter_anomaly_detection(data_path):
    data = get_data(data_path)
    # 异常检测
    data['date'] = data['timestamp'].apply(lambda i: datetime.fromtimestamp(i))
    anomalies = detect_ts(data[['date', 'value']], max_anoms=0.001, direction='both')
    print(anomalies)

    # 时间转换

    plt.plot(pd.to_datetime(data['date']), data['value'], '-')

    # 绘制实际异常散点图
    date = data.loc[data['label'] == 1]['date']
    value = data.loc[data['label'] == 1]['value']
    plt.scatter(pd.to_datetime(date), value, c='b', linewidths=3)

    # 绘制检测结果
    anoms_date = anomalies['anoms']['timestamp']
    plt.plot(pd.to_datetime(pd.to_datetime(anoms_date), format="%Y%m%d %H:%M:%S"), anomalies['anoms']['anoms'], 'ro')

    plt.grid(True)
    # 保存图片
    plt.savefig(img_path + "twitter_anomaly_detection.png", dpi=1000)
    plt.show()


def arundo_adtk(data_path):
    data = get_data(data_path)
    data['date'] = data['timestamp'].apply(lambda i: datetime.fromtimestamp(i))   # 时间转换
    s_train = data[['date', 'value']]

    # 设置索引项
    s_train = s_train.set_index('date')
    s_train = validate_series(s_train)
    print(s_train)
    # plot(s_train)

    # STL分解+离群点检测
    steps = [
        ("deseasonal", STLDecomposition(freq=20)),
        ("quantile_ad", QuantileAD(high=0.9997, low=0.005))
    ]
    pipeline = Pipeline(steps)
    anomalies = pipeline.fit_detect(s_train)
    print(anomalies)
    # plot(s_train, anomaly_pred=anomalies, ap_color='red', ap_marker_on_curve=True)

    # 绘制检测结果]
    known_anomalies = data.loc[data['label'] == 1]
    known_anomalies = known_anomalies[['date', 'label']]
    known_anomalies = known_anomalies.set_index('date')
    known_anomalies = to_events(known_anomalies)
    print(known_anomalies)
    plot(s_train, anomaly_true=known_anomalies, anomaly_pred=anomalies, ap_color='red', ap_marker_on_curve=True,
         at_color="orange")

    plt.savefig(img_path + "arundo_adtk.png", dpi=1000)
    plt.show()


def linkin_luminol(data_path):
    data = get_data(data_path)
    data = data[['timestamp', 'value']]
    # 异常检测
    detector = AnomalyDetector(data)
    anomalies = detector.get_anomalies()
    print(anomalies)

    # 绘制实际异常散点图
    date = data.loc[data['label'] == 1]['timestamp']
    value = data.loc[data['label'] == 1]['value']
    data['date'] = data['timestamp'].apply(lambda i: datetime.fromtimestamp(i))  # 时间转换
    plt.scatter(pd.to_datetime(date), value, c='b', linewidths=3)

    # 绘制检测结果
    plt.plot(pd.to_datetime(anomalies['anoms']['timestamp'], format="%Y%m%d %H:%M:%S"), anomalies['anoms']['anoms'], 'ro')

    plt.grid(True)
    # 保存图片
    plt.savefig(img_path + "twitter_anomaly_detection.png", dpi=1000)
    plt.show()


if __name__ == '__main__':
    # twitter_anomaly_detection(real_data_path+"0.csv")
    # arundo_adtk(real_data_path+"0.csv")
    linkin_luminol(real_data_path+"0.csv")