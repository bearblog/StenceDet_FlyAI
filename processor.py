# -*- coding: utf-8 -*
import os
import sys

from flyai.processor.base import Base
from transformers import BertTokenizer
# from keras.preprocessing.sequence import pad_sequences

from data_helper import data_clean, truncate_and_pad
# import bert.tokenization as tokenization
# from bert import tokenization
# import jieba
# from bert.run_classifier import convert_single_example_simple


class Processor(Base):
    def __init__(self):
        self.tokenizer = None

    def input_x(self, TARGET, TEXT):
        """
        参数为csv中作为输入x的一条数据，该方法会被Dataset多次调用
        multi_cased_L-12_H-768_A-12 bert模型的一个种类
        chinese_roberta_wwm_large_ext_L-24_H-1024_A-16
        chinese_L-12_H-768_A-12
        """
        if self.tokenizer is None:
            # Load pre-trained model tokenizer (vocabulary)
            bert_vocab_file = os.path.join(sys.path[0], 'data/input/model/', 'vocab.txt')
            self.tokenizer = BertTokenizer.from_pretrained(bert_vocab_file)

        # max_seq_length=256
        TEXT = data_clean(TEXT)
        tokenized_id, seq_mask = truncate_and_pad(TEXT, TARGET, 160, self.tokenizer)
        return tokenized_id, seq_mask, TARGET+"\t"+TEXT

    def input_y(self, STANCE):
        """
        参数为csv中作为输入y的一条数据，该方法会被Dataset多次调用
        """
        if STANCE == 'NONE':
            return 0
        elif STANCE == 'FAVOR':
            return 1
        elif STANCE == 'AGAINST':
            return 2

    def output_y(self, data):
        """
        验证时使用，把模型输出的y转为对应的结果
        """
        return data[0]
