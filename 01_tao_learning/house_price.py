import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data
from utils import Accumulator

from sklearn.model_selection import train_test_split

os.chdir(sys.path[0])

def load_data(batch_size):
    df_train = pd.read_csv(
        "./data/kaggle_house_price/kaggle_house_pred_train_processed.csv",
        sep="\t")
    training_data, validate_data = train_test_split(df_train, test_size=0.2, random_state=25)

    X, y = training_data.iloc[:, :-1].to_numpy(), training_data.iloc[:, -1].to_numpy()
    X2, y2 = validate_data.iloc[:, :-1].to_numpy(), validate_data.iloc[:, -1].to_numpy()
    # print(X.shape, y.shape, X2.shape, y2.shape)
    X, y = torch.from_numpy(X).to(torch.float32), torch.from_numpy(y.reshape((-1,1))).to(torch.float32)
    X2, y2 = torch.from_numpy(X2).to(torch.float32), torch.from_numpy(y2.reshape((-1,1))).to(torch.float32)
    dataset_train = data.TensorDataset(X, y)
    dataset_valid = data.TensorDataset(X2, y2)
    train_iter = data.DataLoader(dataset_train, batch_size, shuffle=True, num_workers=1)
    valid_iter = data.DataLoader(dataset_valid, batch_size, shuffle=False, num_workers=1)
    # for X, y in train_iter:
    #     print(X.shape, y.shape)
    #     break
    return train_iter, valid_iter

def RMSELoss(yhat,y):
    return (yhat/y-1)**2

def gen_net():
    def init_weights(m):
        if type(m) == nn.Linear:
            nn.init.normal_(m.weight, std=0.01)

    net = nn.Sequential(nn.Linear(206, 1))
    net.apply(init_weights)
    return net

def calc_rse(y_hat, y):
    rse = float(abs(y_hat/y-1).sum())
    return rse


def train_epoch(net, train_iter, loss, updater):
    if isinstance(net, torch.nn.Module):
        net.train()

    metric = Accumulator(3)
    for X, y in train_iter:
        y_hat = net(X)
        l = loss(y_hat, y)
        # total_loss += l
        if isinstance(updater, torch.optim.Optimizer):
            updater.zero_grad()
            l.mean().backward()
            updater.step()
        rse = calc_rse(y_hat, y)
        metric.add(l.sum(), rse, y.numel())
    return metric[0]/metric[2], metric[1]/metric[2]

def evaluate_rse(net, data_iter):
    if isinstance(net, torch.nn.Module):
        net.eval()
    metric = Accumulator(2)
    with torch.no_grad():
        y_hat_list, y_list = [], []
        for X, y in data_iter:
            y_hat = net(X)
            metric.add(calc_rse(y_hat, y), y.numel())
    return metric[0] / metric[1]


def train(net, train_iter, test_iter, loss, num_epochs, updater):  #@save
    """训练模型（定义见第3章）"""
    for epoch in range(num_epochs):
        train_metrics = train_epoch(net, train_iter, loss, updater)
        test_rse = evaluate_rse(net, test_iter)
        print(f"epoch:{epoch}: loss:{train_metrics[0]:.5f}, train_rse:{train_metrics[1]:.1%}, test_rse:{test_rse:.1%}") 


if __name__ == "__main__":
    batch_size = 10
    lr = 0.003
    train_iter, valid_iter = load_data(batch_size)
    net = gen_net()
    loss = RMSELoss
    optimizer = torch.optim.SGD(net.parameters(), lr=lr)

    num_epochs = 10
    train(net, train_iter, valid_iter, loss, num_epochs, optimizer)

    
