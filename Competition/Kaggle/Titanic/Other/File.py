# -*- coding: utf-8 -*-
# @Time    : 2019-07-02 21:37
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : File.py


def store_csv(data, file_path, date_format="%Y%m%d", index=None, encoding="utf-8", header=True):
    data.to_csv(path_or_buf=file_path, date_format=date_format, index=index, encoding=encoding,
                header=header)
