# -*- coding: utf-8 -*-
# @time       : 1/12/2019 9:04 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @description:

from datetime import datetime

from TimeSerialAnalysis.KPI.config import origin_data_path, real_data_path
from TimeSerialAnalysis.KPI.tool import get_data, store_data


def original_data_classification_by_kpi_id():
    origin_data = get_data(origin_data_path)
    groups_data = origin_data.groupby('KPI ID')
    for group_data in groups_data:
        store_data(group_data[1][['timestamp', 'value', 'label']], real_data_path + group_data[0][0] + ".csv", sep=",",
                   index=False, header=['timestamp', 'value', 'label'])
    return origin_data


if __name__ == '__main__':
    original_data_classification_by_kpi_id()
