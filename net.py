# -*- coding: utf-8 -*
import os
import sys
import torch
from torch import nn
from torchsummary import summary
from transformers import BertModel, BertConfig
from path import BERT_CONFIG_PATH, BERT_MODEL_PATH

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        modelConfig = BertConfig.from_pretrained(BERT_CONFIG_PATH)
        # https://huggingface.co/transformers/v2.4.0/model_doc/bert.html#transformers.BertModel
        self.bert = BertModel.from_pretrained(BERT_MODEL_PATH, config=modelConfig)
        self.out = nn.Linear(768, 3)

    def forward(self, input):
        tokens_tensor, segments_tensors = input
        # with torch.no_grad():
        outputs = self.bert(tokens_tensor)
        cls = outputs[1]

        output = self.out(cls)
        return output


if __name__ == "__main__":
    net = Net()
    summary(net, (8, 128))
    # print(net)