# -*- coding: utf-8 -*-
# @time       : 15/11/2019 3:33 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @description: 

import sys, os
sys.path.append("..")
sys.path.append(os.path.abspath("../../"))


from flask import Flask, render_template, request, make_response
from flask import jsonify

import sys
import time
import hashlib
import threading
import jieba
from annoy import AnnoyIndex

from chatbot import calculate_top_k
from conf.path_config import boasting_path, matrix_ques_xlnet_save_path, matrix_ques_bert_save_path
from utils.text_tools import txtRead
from feature.bert.extract_keras_bert_feature import KerasBertVector
from feature.xlnet.extract_keras_xlnet_feature import KerasXlnetVector

from tqdm import tqdm

n1 = 768
n2 = 768


def heartbeat():
    print(time.strftime('%Y-%m-%d %H:%M:%S - heartbeat', time.localtime(time.time())))
    timer = threading.Timer(60, heartbeat)
    timer.start()


timer = threading.Timer(60, heartbeat)
timer.start()


app = Flask(__name__, static_url_path="/static")

bert_annoy_index = AnnoyIndex(n1, 'angular')
bert_annoy_index.load(matrix_ques_bert_save_path)

xlnet_annoy_index = AnnoyIndex(n2, 'angular')
xlnet_annoy_index.load(matrix_ques_xlnet_save_path)

questions = txtRead(boasting_path, encodeType='utf-8')
ans = []
for ques in tqdm(questions):
    ans.append(ques.split('\t')[1])
print("load annoy index completed!")
bert_vector = KerasBertVector()
xlnet_vector = KerasXlnetVector()
print('load model completed!')


@app.route('/message', methods=['POST'])
def reply():
    req_msg = request.form['msg']
    res_msg = calculate_top_k(req_msg, bert_annoy_index, bert_vector, xlnet_annoy_index, xlnet_vector, questions, ans, k=1)
    return jsonify({'text': res_msg})


@app.route("/")
def index():
    return render_template("index.html")


if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8808)
