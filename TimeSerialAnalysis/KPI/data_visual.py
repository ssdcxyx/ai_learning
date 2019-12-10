# -*- coding: utf-8 -*-
# @time       : 10/12/2019 8:30 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @description: 

import matplotlib
import matplotlib.pyplot as plt
import os
import pandas as pd
from matplotlib.dates import DateFormatter
import cv2
from datetime import datetime

from TimeSerialAnalysis.KPI.config import real_data_path, img_path
from TimeSerialAnalysis.KPI.tool import get_data


def visual_all_data(data_path):
    files = os.listdir(data_path)
    for (i, file) in zip(range(len(files)), files):
        data = get_data(data_path + file)
        data['date'] = data['timestamp'].apply(lambda i: datetime.fromtimestamp(i))
        plt.plot(data['date'], data['value'], '-', )
        for i in range(2):
            date = data.loc[data['label'] == 1]['date']
            value = data.loc[data['label'] == 1]['value']
            plt.scatter(date, value, c='r')
        plt.title(file)
    plt.savefig(img_path + "visual_all_data.png", dpi=1000)
    plt.grid(True)
    plt.show()


def visual_data(data_path):
    data = get_data(data_path)
    data['date'] = data['timestamp'].apply(lambda i: datetime.fromtimestamp(i))
    plt.plot(data['date'], data['value'], '-', )
    date = data.loc[data['label'] == 1]['date']
    value = data.loc[data['label'] == 1]['value']
    plt.scatter(date, value, c='r')
    plt.title(data_path)
    plt.savefig(img_path + "visual_data.png", dpi=1000)
    plt.grid(True)
    plt.show()


if __name__ == '__main__':
    # visual_all_data(data_path=real_data_path)
    visual_data(data_path=real_data_path+"0.csv")