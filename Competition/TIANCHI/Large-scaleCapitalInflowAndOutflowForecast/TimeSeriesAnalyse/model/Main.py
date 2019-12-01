# -*- coding: utf-8 -*-
# @Time    : 2019-05-17 20:41
# @Author  : ssdcxy
# @Email   : 18379190862@163.com
# @File    : Main.py

from han.GetData import get_data
from handle import DataProcess
from setting import Constant
from model import Model
from handle import DatVisual
from handle import File

import pandas as pd


if __name__ == '__main__':
    user_classification_data = get_data(file_name=Constant.user_classification_data_table)
    labels = Model.kmeans(user_classification_data)
    tsne = Model.tsne(user_classification_data)
    DatVisual.show_classification_distribution(tsne, target=labels)
    pca = Model.pca(user_classification_data)
    DatVisual.show_classification_distribution(pca, target=labels)
    labels_df = pd.DataFrame({"label": labels})
    classification_result = pd.concat([user_classification_data, labels_df], axis=1)
    DatVisual.show_kinds_of_user_data(classification_result)
    File.store_csv(classification_result, file_path=Constant.user_classification_result_table)
    print()
