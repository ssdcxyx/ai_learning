# -*- coding: utf-8 -*-
# @time       : 10/12/2019 8:31 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @description: 

import pandas as pd


def get_data(data_path, sep=",", date_parser=None):
    return pd.read_csv(data_path, sep=sep,  date_parser=date_parser)


def store_data(data, data_path, sep=",", index=False, header=None):
    return data.to_csv(data_path, sep=sep, index=index, header=header)


