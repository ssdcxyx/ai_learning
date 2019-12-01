# -*- coding: utf-8 -*-
# @time       : 15/11/2019 3:04 下午
# @author     : ssdcxy
# @email      : 18379190862@163.com
# @description:

import json
from tqdm import tqdm


def json_preprocess(input_path, output_path):
    print("read json data...")
    _input = open(input_path, "r", encoding="utf-8")
    qas = []
    for line in tqdm(_input.readlines()):
        data = json.loads(line)
        qas.append(data)
    _output = open(output_path, "w", encoding="utf-8")
    print("write qa txt data...")
    for qa in tqdm(qas):
        question = qa['question'].strip('\t').strip("\n")
        answer = qa['answers'][0].strip("\t").strip("\n")
        _output.write(question+"\t"+answer + "\n")
    _input.close()
    _output.close()
    print("process data done")
    return qas
