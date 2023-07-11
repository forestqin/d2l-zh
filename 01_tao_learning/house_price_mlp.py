import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data
import utils

from sklearn.model_selection import train_test_split

os.chdir(sys.path[0])

def load_data():
    df_train = pd.read_csv(
        "./data/kaggle_house_price/kaggle_house_pred_train_processed.csv",
        sep="\t")
    train_features = torch.tensor(df_train.iloc[:, :-1].values, dtype=torch.float32)
    train_labels = torch.tensor(df_train.iloc[:, -1].values.reshape(-1, 1), dtype=torch.float32)
    return train_features, train_labels

def load_data2():
    df_train = pd.read_csv(
        "./data/kaggle_house_price/kaggle_house_pred_train_processed.csv",
        sep="\t")
    training_data, validate_data = train_test_split(df_train, test_size=0.2, random_state=25)
    
    train_X = torch.tensor(training_data.iloc[:, :-1].values, dtype=torch.float32)
    train_y = torch.tensor(training_data.iloc[:, -1].values.reshape(-1, 1), dtype=torch.float32)

    valid_X = torch.tensor(validate_data.iloc[:, :-1].values, dtype=torch.float32)
    valid_y = torch.tensor(validate_data.iloc[:, -1].values.reshape(-1, 1), dtype=torch.float32)

    return train_X, train_y, valid_X, valid_y


def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


def load_array(data_arrays, batch_size, is_train=True):
    dataset = data.TensorDataset(*data_arrays)
    return data.DataLoader(dataset, batch_size, shuffle=is_train)


def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = load_array((train_features, train_labels), batch_size)
    # 这里使用的是Adam优化算法
    optimizer = torch.optim.Adam(net.parameters(),
                                 lr = learning_rate,
                                 weight_decay = weight_decay)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            optimizer.zero_grad()
            l = loss(net(X), y)
            l.backward()
            optimizer.step()
        train_rmse = log_rmse(net, train_features, train_labels)
        train_ls.append(train_rmse)
        if test_labels is not None:
            test_rsme = log_rmse(net, test_features, test_labels)
            test_ls.append(test_rsme)
        if epoch % 100 == 0:
            print(f"epoch:{epoch}: train_rsme:{train_rmse:.1%}, test_rsme:{test_rsme:.1%}") 
    return train_ls, test_ls


def get_net():
    # net = nn.Sequential(nn.Linear(206,32), nn.ReLU(), nn.Dropout(0.1), nn.Linear(32, 1))
    net = nn.Sequential(nn.Linear(206, 16), nn.ReLU(), nn.Linear(16, 1))
    # net = nn.Sequential(nn.Linear(206, 1))
    return net


loss = nn.MSELoss()

if __name__ == "__main__":
    num_epochs, lr, weight_decay, batch_size = 1000, 5, 5, 64
    train_X, train_y, valid_X, valid_y = load_data2()
    net = get_net()
    train_ls, valid_ls = train(net, train_X, train_y, valid_X, valid_y, num_epochs, lr, weight_decay, batch_size)
    print(f'训练log rmse{float(train_ls[-1]):.1%}, 'f'验证log rmse{float(valid_ls[-1]):.1%}')