# -*- coding: utf-8 -*
import sys
import os

from flyai.utils import remote_helper

# 训练数据的路径
DATA_PATH = os.path.join(sys.path[0], 'data', 'input')
# 模型保存的路径
MODEL_PATH = os.path.join(sys.path[0], 'data', 'output', 'model')
# 训练log的输出路径
LOG_PATH = os.path.join(sys.path[0], 'data', 'output', 'logs')

# 下载预训练模型
remote_helper.get_remote_date('https://www.flyai.com/m/chinese_wwm_pytorch.zip')
# bert_path = os.path.splitext(path)[0]
# BERT模型文件
BERT_CONFIG_PATH = os.path.join(sys.path[0], 'data/input/model/bert_config.json')
BERT_MODEL_PATH = os.path.join(sys.path[0], 'data/input/model/pytorch_model.bin')