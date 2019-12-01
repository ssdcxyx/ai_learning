# -*- coding: utf-8 -*-
# @time       : 15/11/2019 11:12 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @description:

from feature.bert.extract_keras_bert_feature import KerasBertVector
from feature.xlnet.extract_keras_xlnet_feature import KerasXlnetVector
from conf.path_config import *
from utils.text_tools import txtRead
import numpy as np
from annoy import AnnoyIndex


def store_ann(ques_basic_vecs, output_path):
    m, n = ques_basic_vecs.shape
    print("n size is:", n)
    t = AnnoyIndex(n, 'angular')
    for i in range(m):
        t.add_item(i, ques_basic_vecs[i])

    t.build(100)
    t.save(output_path)


def calculate_sentence_vec_by_own():
    questions = txtRead(boasting_path, encodeType='utf-8')
    ques = [ques.split('\t')[0] for ques in questions]
    # 生成标准问题的bert句向量
    bert_vector = KerasBertVector()
    ques_basic_bert_vecs = bert_vector.bert_encode(ques)
    # 生成标准标题的xlnet向量
    xlnet_vector = KerasXlnetVector()
    ques_basic_xlnet_vecs = xlnet_vector.xlnet_encode(ques)

    store_ann(np.array(ques_basic_bert_vecs), output_path=matrix_ques_bert_save_path)
    store_ann(np.array(ques_basic_xlnet_vecs), output_path=matrix_ques_xlnet_save_path)


def calculate_top_k(query, bert_annoy_index, bert_vector, xlnet_annoy_index, xlnet_vector, questions, ans,  k=1):

    query_bert_vec = bert_vector.bert_encode([query])[0]
    query_bert_vec = np.array(query_bert_vec)

    query_xlnet_vec = xlnet_vector.xlnet_encode([query])[0]
    query_xlnet_vec = np.array(query_xlnet_vec)
    topk = []
    topk_bert = list(bert_annoy_index.get_nns_by_vector(query_bert_vec, k, search_k=-1, include_distances=True))
    topk.extend(topk_bert)
    topk_xlnet = list(xlnet_annoy_index.get_nns_by_vector(query_xlnet_vec, k, search_k=-1, include_distances=True))
    topk[0].extend(topk_xlnet[0])
    topk[1].extend(topk_xlnet[1])
    topk = zip(topk[0], topk[1])
    topk = sorted(topk, key=lambda x: (x[1], x[0]))
    topk = zip(*topk)
    topk_index, topk_dis = [list(x) for x in topk]
    _result = list(set(topk_index))
    _ans = []
    for topk_index in _result:
        _ans.append(ans[topk_index])
    return _ans


if __name__ == "__main__":
    calculate_sentence_vec_by_own()

