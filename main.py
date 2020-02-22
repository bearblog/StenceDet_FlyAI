# -*- coding: utf-8 -*-
"""
Created on Mon Oct 30 19:44:02 2017

@author: Sherlock
"""

import argparse

import torch
import torch.nn as nn
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup
from flyai.dataset import Dataset
from flyai.utils.log_helper import train_log

from model import Model
from net import Net
from path import MODEL_PATH
'''
样例代码仅供参考学习，可以自己修改实现逻辑。
Tensorflow模版项目下载： https://www.flyai.com/python/tensorflow_template.zip
PyTorch模版项目下载： https://www.flyai.com/python/pytorch_template.zip
Keras模版项目下载： https://www.flyai.com/python/keras_template.zip
第一次使用请看项目中的：第一次使用请读我.html文件
常见问题请访问：https://www.flyai.com/question
意见和问题反馈有红包哦！添加客服微信：flyaixzs
'''

if __name__ == "__main__":

    '''
    项目的超参
    '''
    parser = argparse.ArgumentParser()
    parser.add_argument("-e", "--EPOCHS", default=5, type=int, help="train epochs")
    parser.add_argument("-b", "--BATCH", default=16, type=int, help="batch size")
    args = parser.parse_args()

    '''
    flyai库中的提供的数据处理方法
    传入整个数据训练多少轮，每批次批大小
    '''
    dataset = Dataset(epochs=args.EPOCHS, batch=args.BATCH)
    print(f'train datas: {dataset.get_train_length()}, val data: {dataset.get_validation_length()}')
    lr = 1e-4
    num_warmup_steps = 1000
    max_grad = 1.0
    '''
    实现自己的网络机构
    '''
    # 判断gpu是否可用
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    net = Net().to(device)
    model = Model(dataset, net)
    # print(net)
    # optimizer = torch.optim.Adam(net.parameters(), lr=5e-6)
    optimizer = AdamW(net.parameters(), lr=lr, correct_bias=False)
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=dataset.get_step())
    criterion = nn.CrossEntropyLoss()
    ''' 
    dataset.get_step() 获取数据的总迭代次数
    
    '''
    total_loss, train_acc, best_score = 0., 0., 0.
    train_bar = tqdm(range(dataset.get_step()), desc='Iteration')
    for step in train_bar:
        net.train()
        x_train, y_train = dataset.next_train_batch()
        x_val, y_val = dataset.next_validation_batch()
        x_train = [torch.from_numpy(data).to(device) for data in x_train[:-1]]
        y_train = torch.from_numpy(y_train).to(device)
        '''
        实现自己的模型保存逻辑
        '''
        # initialise gradients
        optimizer.zero_grad()
        # generate predictions
        outputs = net(x_train)
        # calculate loss
        loss = criterion(outputs, y_train)
        # compute loss
        loss.backward()
        nn.utils.clip_grad_norm_(net.parameters(), max_grad)
        # update parameters using gradients
        optimizer.step()
        scheduler.step()

        total_loss += loss.item()
        train_acc_ts = (outputs.argmax(1) == y_train)
        train_acc += (train_acc_ts).sum().item()
        train_bar.set_description('{}/{} loss: {:.4f}'.format(step + 1, dataset.get_step(), loss.item()))
        if step % 20 == 0:
            data_num = 20 * args.BATCH
            val_loss, val_acc = model.evaluate()
            train_log(total_loss/data_num, train_acc/data_num, val_loss, val_acc)
            if val_acc > best_score:
                model.save_model(net, MODEL_PATH, overwrite=True)
                best_score = val_acc
                print("Model saved!")
            total_loss, train_acc = 0., 0.
