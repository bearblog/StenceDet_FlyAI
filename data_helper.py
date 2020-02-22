# -*- coding: utf-8 -*-
# @Time : 2019/5/30 下午5:53
# @Author : dabing
# @File : data_helper.py
# @Software: PyCharm

import os
import re
import sys
import jieba

# import numpy as np
import pandas as pd

def data_clean(sent):
    """
    1. 数据清洗
    """
    #    if len(sent) == 0:
    #        print('[ERROR] data_clean faliled! | The params: {}'.format(sent))
    #        return None
    ##    unicode 编码中，中文范围为4e00-9fa5
    #    sentence = re.sub('[^\u4e00-\u9fa5]', ' ', sent).strip().replace('  ', ' ', 3)
    if len(sent) == 0:
        print('[ERROR] data_clean faliled! | The params: {}'.format(sent))
        return None
    #    sentence = re.sub(r'[【】', ' ', sent).strip().replace('  ', ' ')
    sentence = regular(sent)
    #    sentence = remove_stop_words(sentence)
    return sentence


def regular(text):
    # 去除(# #)字段
    text = re.sub(r'#.*#', ' ', text)
    # 去除多个@用户
    text = re.sub(r'@([\u4e00-\u9fa5a-zA-Z0-9_-]+)', ' ', text)
    # 去除url
    text = re.sub(r'http[s]?://[a-zA-Z0-9.?/&=:]*', ' ', text)
    # 去除英文和数字
    #    text = re.sub(r'[a-zA-Z0-9]', ' ', text)
    # 去除其他噪音字符
    text = re.sub(r'[—|（）()【】…「~_]+', ' ', text)
    # 去除多余空格
    text = re.sub(r'\s+', ' ', text)
    # 去除首行空格
    text = text.strip()

    return text


def remove_stop_words(text):
    new_text = []
    stop_words = []
    path = os.path.join(sys.path[0], 'stop_words.txt')
    with open(path, 'r', encoding='utf-8') as file:
        for line in file.readlines():
            stop_words.append(line.strip())
    word_segment = jieba.lcut(text)

    for word in word_segment:
        if word not in stop_words:
            new_text.append(word)

    return ''.join(new_text)


def truncate_and_pad(text, target, max_seq_len, tokenizer):
    """
    对输入文本进行分词、截断和填充
    :param text:
    :param target:
    :param max_seq_len:
    :param tokenizer:
    :return:
    """
    text = '[CLS]' + text + '[SEP]' + target + '[SEP]'
    tokenized_sentence = tokenizer.tokenize(text)
    tokenized_id = tokenizer.convert_tokens_to_ids(tokenized_sentence)
    # tokenized_id = pad_sequences(tokenized_id, maxlen=128, dtype="long", truncating="post", padding="post")

    if len(tokenized_id) < max_seq_len:
        tokenized_id += [0] * (max_seq_len - len(tokenized_id))
    else:
        tokenized_id = tokenized_id[:max_seq_len]
    seq_mask = [float(i > 0) for i in tokenized_id]
    return tokenized_id, seq_mask


if __name__ == "__main__":
    # text = '#深圳禁摩限电# 自行车、汽车也同样会引发交通事故——为何单怪他们？（我早就发现：交通的混乱，反映了“交管局”内部的混乱！）必须先严整公安交管局内部！——抓问题的根本！@深圳交警@中国政府网@人民日报'
    # print(data_clean(text))
    data = pd.read_csv("data/input/dev.csv")
    print(data['TEXT'].apply(len).describe())
