# -*- coding: utf-8 -*
import numpy
import os
import torch
import torch.nn as nn

from tqdm import tqdm
from flyai.model.base import Base

from path import MODEL_PATH

__import__('net', fromlist=["Net"])

TORCH_MODEL_NAME = "model.pkl"


class Model(Base):
    def __init__(self, data, net=None):
        self.data = data
        self.net_path = os.path.join(MODEL_PATH, TORCH_MODEL_NAME)
        if os.path.exists(self.net_path):
            self.net = torch.load(self.net_path)
        if net:
            self.net = net
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.criterion = nn.CrossEntropyLoss()

    def evaluate(self):
        self.net.eval()
        val_acc, val_loss = 0., 0.
        batch = 32
        val_num = self.data.get_validation_length()
        # val_bar = tqdm(range(self.data.get_step()), desc='Iteration')
        with torch.no_grad():
            for step in range(self.data.get_step()):
                if step * batch > val_num:
                    break
                x_val, y_val = self.data.next_validation_batch()
                x_val_text = x_val[-1]
                x_val = [torch.from_numpy(data).to(self.device) for data in x_val[:-1]]
                y_val = torch.from_numpy(y_val).to(self.device)

                # generate predictions
                outputs = self.net(x_val)
                train_acc_ts = outputs.argmax(1) == y_val
                for id, item in enumerate(train_acc_ts.cpu().numpy()):
                    if item == 0:
                        print("{}\t{}".format(x_val_text[id], y_val.cpu().numpy()[id]))
                val_loss += self.criterion(outputs, y_val).item()
                val_acc += (train_acc_ts).sum().item()
        val_loss = val_loss / (step * batch)
        val_acc = val_acc / (step * batch)
        return val_loss, val_acc

    def predict(self, **data):
        if self.net is None:
            self.net = torch.load(self.net_path)
        x_data = self.data.predict_data(**data)
        # x_data = torch.from_numpy(x_data)
        x_data = [torch.from_numpy(data).to(self.device) for data in x_data]
        outputs = self.net(x_data)
        prediction = torch.argmax(outputs, dim=1).data.cpu().numpy()
        prediction = self.data.to_categorys(prediction)
        return prediction

    def predict_all(self, datas):
        if self.net is None:
            self.net = torch.load(self.net_path)
        labels = []
        for data in datas:
            x_data = self.data.predict_data(**data)
            # x_data = torch.from_numpy(x_data)
            x_data = [torch.from_numpy(data).to(self.device) for data in x_data]
            outputs = self.net(x_data)
            # 将概率值转换为预测标签
            prediction = torch.argmax(outputs, dim=1).data.cpu().numpy()
            prediction = self.data.to_categorys(prediction)
            labels.append(prediction)
        return labels

    def batch_iter(self, x, y, batch_size=128):
        """生成批次数据"""
        data_len = len(x)
        num_batch = int((data_len - 1) / batch_size) + 1

        indices = numpy.random.permutation(numpy.arange(data_len))
        x_shuffle = x[indices]
        y_shuffle = y[indices]

        for i in range(num_batch):
            start_id = i * batch_size
            end_id = min((i + 1) * batch_size, data_len)
            yield x_shuffle[start_id:end_id], y_shuffle[start_id:end_id]

    def save_model(self, network, path, name=TORCH_MODEL_NAME, overwrite=False):
        super().save_model(network, path, name, overwrite)
        torch.save(network, os.path.join(path, name))