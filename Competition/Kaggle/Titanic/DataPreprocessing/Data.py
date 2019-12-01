# -*- coding: utf-8 -*-
# @Time    : 2019-06-29 11:04
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Data.py
import pandas as pd


def get_data(file_path, encoding='utf-8', parse_dates=None, date_parser=None):
    if parse_dates is None:
        return pd.read_csv(file_path, encoding=encoding)
    else:
        return pd.read_csv(file_path, encoding=encoding, parse_dates=parse_dates, date_parser=date_parser)


if __name__ == '__main__':
    print()
