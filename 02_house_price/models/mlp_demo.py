import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data

os.chdir(sys.path[0])


def process2(all_features):
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True)
    print(all_features.shape)
    return all_features


def get_feature_df(train_input, test_input):
    train = pd.read_csv(train_input)
    test = pd.read_csv(test_input)
    train_data = train.drop(['Id', 'SalePrice'], axis=1)
    test_data = test.drop(['Id'], axis=1)
    all_features = pd.concat((train_data.iloc[:, :], test_data.iloc[:, :]))
    
    # feature_df = process(all_features)
    feature_df = process2(all_features)

    n_train = train.shape[0]
    train_df = torch.tensor(feature_df[:n_train].values, dtype=torch.float32)
    train_label = torch.tensor(train.SalePrice.values.reshape(-1, 1), dtype=torch.float32)
    test_df  = torch.tensor(feature_df[n_train:].values, dtype=torch.float32)
    test_Id = test['Id']
    feature_list = feature_df.columns.tolist()


    print(f'Training Shape: {train_df.shape}, {train_label.shape}')
    print(f'Testing  Shape: {test_df.shape}, {test_Id.shape}')

    return train_df, train_label, test_df, test_Id, feature_list

def log_rmse(net, features, labels):
    # 为了在取对数时进一步稳定该值，将小于1的值设置为1
    clipped_preds = torch.clamp(net(features), 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds),
                           torch.log(labels)))
    return rmse.item()


def gen_dataloader(X, y, batch_size):
    dataset = data.TensorDataset(X, y)
    data_iter = data.DataLoader(dataset, batch_size, shuffle=True)
    return data_iter

def train(net, train_features, train_labels, test_features, test_labels,
          num_epochs, learning_rate, weight_decay, batch_size):
    train_ls, test_ls = [], []
    train_iter = gen_dataloader(train_features, train_labels, batch_size)
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
            test_rmse = log_rmse(net, test_features, test_labels)
            test_ls.append(test_rmse)
            print(f"{epoch=}: {train_rmse=}, {test_rmse=}")
    return train_ls, test_ls


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = None, None
    for j in range(k):
        idx = slice(j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X[idx, :], y[idx]
        if j == i:
            X_valid, y_valid = X_part, y_part
        elif X_train is None:
            X_train, y_train = X_part, y_part
        else:
            X_train = torch.cat([X_train, X_part], 0)
            y_train = torch.cat([y_train, y_part], 0)
    return X_train, y_train, X_valid, y_valid


train_input = "../data/kaggle_house_price/kaggle_house_pred_train.csv"
test_input = "../data/kaggle_house_price/kaggle_house_pred_test.csv"
train_features, train_labels, test_features, test_Id, feature_list = get_feature_df(train_input, test_input)
k, num_epochs, learning_rate, weight_decay, batch_size = 4, 100, 5, 0, 64
in_features = train_features.shape[1]
net = nn.Sequential(nn.Linear(in_features,1))
loss = nn.MSELoss()

X_train, y_train, X_valid, y_valid = get_k_fold_data(k, 0, train_features, train_labels)

train(net, X_train, y_train, X_valid, y_valid, num_epochs, learning_rate, weight_decay, batch_size)
