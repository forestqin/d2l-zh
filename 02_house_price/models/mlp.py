import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data

import utils
from config import log
from sklearn.model_selection import train_test_split
device = 'cuda' if torch.cuda.is_available() else 'cpu'

os.chdir(sys.path[0])

class MLPModel:

    def __init__(self, feature_list):
        self.name = "MLP"
        self.feature_list = feature_list
        self.model = None
        self.trained = False

        self.num_epochs = 50
        self.batch_size = 64
        self.lr = 5
        self.weight_decay = 5

    def gen_dataloader(self, X, y, is_train):
        X = torch.tensor(X.values, dtype=torch.float32)
        y = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32)
        dataset = data.TensorDataset(X, y)
        data_iter = data.DataLoader(dataset, self.batch_size, shuffle=is_train)
        return data_iter, X, y

    def train(self, X, y, val_X, val_y):
        # 1. get data
        train_iter, X, y = self.gen_dataloader(X, y, True)
        val_iter, val_X, val_y = self.gen_dataloader(val_X, val_y, False)

        # 2. build model
        input_num = len(self.feature_list)
        model = nn.Sequential(nn.Linear(input_num, 16), nn.ReLU(), nn.Linear(16, 1))
        loss = nn.MSELoss()
        # optimizer = torch.optim.Adam(model.parameters(), self.lr, self.weight_decay)
        optimizer = torch.optim.SGD(model.parameters(), self.lr, self.weight_decay)
        
        #3. train
        train_ls, test_ls = [], []
        for epoch in range(self.num_epochs):
            for _X, _y in train_iter:
                optimizer.zero_grad()
                l = loss(model(_X), _y)
                l.backward()
                optimizer.step()
            train_rmse = self.log_rmse(X, y)
            train_ls.append(train_rmse)
            if val_y is not None:
                test_rsme = self.log_rmse(val_X, val_y)
                test_ls.append(test_rsme)
            if epoch % 2 == 0:
                log.debug(f"epoch:{epoch}: train_rsme:{train_rmse:.1%}, test_rsme:{test_rsme:.1%}") 
        
        log.debug(f'训练log rmse{float(train_ls[-1]):.1%}, 'f'验证log rmse{float(test_ls[-1]):.1%}')
        
        self.model = model
        self.trained = True

    def log_rmse(self, X, y):
        # 为了在取对数时进一步稳定该值，将小于1的值设置为1
        clipped_preds = torch.clamp(self.model(X), 1, float('inf'))
        rmse = torch.sqrt(loss(torch.log(clipped_preds),
                            torch.log(y)))
        return rmse.item()

    def predict(self, X):
        assert self.model is not None, "model has not been trained"
        y_hat = self.model(X)
        return y_hat

    def submit(self, test_df, test_Id):
        preds = self.predict(test_df)
        submission = pd.DataFrame({"Id": test_Id.values, "SalePrice": preds})
        output_file = f"./output/submission_{self.name}.csv"

        submission.to_csv(output_file, index=False)
        log.info("submission saved to {}".format(output_file))