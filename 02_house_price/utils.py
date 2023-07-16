import pandas as pd
import numpy as np
import time
import datetime
import torch
from torch import nn
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import config
from config import log
from matplotlib import pyplot as plt

class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()

def gen_train_test_dataset(train_df, train_label):
    features = train_df.values
    labels = train_label.values
    train_X, test_X, train_y, test_y = train_test_split(features,
                                                        labels,
                                                        test_size=config.test_size,
                                                        random_state=42)
    log.info(f'Training  Shape: {train_X.shape}, {train_y.shape}')
    log.info(f'Validate  Shape: {test_X.shape}, {test_y.shape}')
    return train_X, test_X, train_y, test_y


def get_k_fold_data(k, i, X, y):
    assert k > 1
    fold_size = X.shape[0] // k
    X_train, y_train = [], []
    for j in range(k):
        a, b = (j * fold_size, (j + 1) * fold_size)
        X_part, y_part = X.iloc[a:b, :], y[a:b]
        if j == i:
            X_valid, y_valid = X_part, y_part
        else:
            X_train.append(X_part)
            y_train.append(y_part)

    X_train = pd.concat(X_train)
    y_train = pd.concat(y_train)

    if i == 0:
        log.info(f'Training  Shape: {X_train.shape}, {y_train.shape}')
        log.info(f'Validate  Shape: {X_valid.shape}, {y_valid.shape}')

    return X_train, y_train, X_valid, y_valid


def calc_score(predictor, X, y):
    y_hat = predictor.predict(X)
    if config.metric == "mape":
        score = np.mean(abs(y_hat / y - 1))*100
    elif config.metric == "rmse":
        score = np.sqrt(mean_squared_error(y, y_hat))
    elif config.metric == "log_rmse":
        score = log_rmse(y_hat, y)
    else:
        raise NotImplementedError
    return score

def log_rmse(y_hat, y):
    # y_hat is numpy, y is dataframe
    loss = nn.MSELoss()
    y = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)
    y_hat = torch.tensor(y_hat.reshape(-1, 1), dtype=torch.float32)
    clipped_preds = torch.clamp(y_hat, 1, float('inf'))
    rmse = torch.sqrt(loss(torch.log(clipped_preds), torch.log(y)))
    return rmse.item()
