# -*- coding: utf-8 -*-
# @time       : 1/12/2019 9:04 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @description: 


import pandas as pd


origin_data_path = "./data/phase2_train.csv"
real_data_path = "./data/real/"


def original_data_classification_by_kpi_id():
    origin_data = pd.read_csv(origin_data_path, sep=",")
    groups_data = origin_data.groupby('KPI ID')
    for group_data in groups_data:
        group_data[1]['timestamp', 'value', 'label'].to_csv(real_data_path + group_data[0][0] + ".csv", sep=",",
                                                            index=False, header=['timestamp', 'value', 'label'])
    return origin_data


if __name__ == "__main__":
    original_data_classification_by_kpi_id()
