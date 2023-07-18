import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data

from config import log
import seaborn as sns
from matplotlib import pyplot as plt


# device = 'cpu'

os.chdir(sys.path[0])

class MLPModel:

    def __init__(self, feature_list):
        self.name = "MLP"
        self.feature_list = feature_list
        self.num_epochs = 3000
        self.interval = 100

        self.batch_size = 128
        self.lr = 0.5
        self.weight_decay = 0.5
        # self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'

        # self.model = nn.Sequential(nn.Linear(len(self.feature_list),32, device=self.device), 
        #                            nn.Dropout(p=0.2), nn.ReLU(), 
        #                            nn.Linear(32, 1, device=self.device))
        self.model = nn.Sequential(nn.Linear(len(self.feature_list), 1, device=self.device))
        log.info(f"{self.device=}")
        log.info(f", {self.model}")

        self.loss = nn.MSELoss()
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                    lr = self.lr,
                                    weight_decay = self.weight_decay)
        self.trained = False
        

    def train(self, X_train, y_train, X_valid, y_valid):
        X_train, y_train = self.transform_tensor(X_train, y_train)
        X_valid, y_valid = self.transform_tensor(X_valid, y_valid)
        train_iter = self.gen_dataloader(X_train, y_train, True)
        
        epoch_ls, train_ls, valid_ls = [], [], []
        for epoch in range(self.num_epochs):
            for X, y in train_iter:
                self.optimizer.zero_grad()
                l = self.loss(self.model(X), y)
                l.backward()
                self.optimizer.step()
            
            if y_valid is not None and epoch % self.interval == 0:
                epoch_ls.append(epoch+1)
                train_rmse = self.log_rmse(X_train, y_train)
                train_ls.append(train_rmse)
                valid_rmse = self.log_rmse(X_valid, y_valid)
                valid_ls.append(valid_rmse)
                log.debug(f"epoch:{epoch}: train_rsme:{train_rmse:.3f}, valid_rmse:{valid_rmse:.3f}") 
        
        # last epoch
        epoch_ls.append(self.num_epochs)
        train_rmse = self.log_rmse(X_train, y_train)
        train_ls.append(train_rmse)
        valid_rmse = self.log_rmse(X_valid, y_valid)
        valid_ls.append(valid_rmse)
        log.info(f'训练log rmse: {float(train_ls[-1]):.3f}, 'f'验证log rmse: {float(valid_ls[-1]):.3f}')
        
        res_df = pd.DataFrame({"epoch":epoch_ls, "train":train_ls, "valid": valid_ls})
        dfm = res_df.melt('epoch', var_name='group', value_name='rmse')
        plt.rcParams['figure.figsize'] = (12, 6)
        sns.pointplot(x="epoch", y="rmse", hue='group', data=dfm)
        # sns.lineplot(x="epoch", y="rmse", hue='group', data=dfm)
        # sns.catplot(x="epoch", y="rmse", hue='group', data=dfm, kind='point')
        plt.show()

        self.trained = True

    def transform_tensor(self, X, y):
        X = torch.tensor(X.values, dtype=torch.float32, device=self.device)
        y = torch.tensor(y.values.reshape(-1, 1), dtype=torch.float32, device=self.device)
        return X, y

    def gen_dataloader(self, X, y, is_train):
        dataset = data.TensorDataset(X, y)
        data_iter = data.DataLoader(dataset, self.batch_size, shuffle=is_train)
        return data_iter

    def log_rmse(self, X, y):
        # 为了在取对数时进一步稳定该值，将小于1的值设置为1
        y_hat  = self.model(X)
        clipped_preds = torch.clamp(y_hat, 1, float('inf'))
        rmse = torch.sqrt(self.loss(torch.log(clipped_preds), torch.log(y)))
        # rmse = torch.sqrt(self.loss(y_hat, y))
        return rmse.item()

    def predict(self, X):
        assert self.model is not None, "model has not been trained"
        X = torch.tensor(X.values, dtype=torch.float32, device=self.device)
        y_hat = self.model(X)
        return y_hat.cpu().detach().numpy().squeeze()

    def submit(self, test_df, test_Id):
        preds = self.predict(test_df)
        submission = pd.DataFrame({"Id": test_Id.values, "SalePrice": preds})
        output_file = f"./output/submission_{self.name}.csv"

        submission.to_csv(output_file, index=False)
        log.info("submission saved to {}".format(output_file))