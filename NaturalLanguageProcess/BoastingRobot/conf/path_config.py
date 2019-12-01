# -*- coding: utf-8 -*-
# @time       : 15/11/2019 11:14 上午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @description: 


import pathlib
import sys
import os

projectdir = str(pathlib.Path(os.path.abspath(__file__)).parent.parent)
sys.path.append(projectdir)

chicken_and_gossip_path = projectdir + '/data/corpus/chicken_and_gossip.txt'
boasting_path = projectdir + '/data/corpus/boasting.txt'

origin_baidu_qa_path = projectdir + '/data/origin/baidu/zhidao_qa.json'
baidu_qa_boasting_path = projectdir + '/data/corpus/baidu_qa.txt'


matrix_ques_bert_save_path = projectdir + '/data/corpus_vector/boasting_bert.ann'
matrix_ques_xlnet_save_path = projectdir + '/data/corpus_vector/boasting_xlnet.ann'
