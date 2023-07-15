import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from scipy.stats import randint
from sklearn.tree import export_graphviz
import pydot
from config import log

os.chdir(sys.path[0])

class RandomForestModel:

    def __init__(self, feature_list):
        self.name = "random_forest"
        self.feature_list = feature_list
        self.model = None
        self.trained = False

    def train(self, X, y):
        assert X.shape[1] == len(self.feature_list), f"数据集{X.shape}不匹配特征数量{len(self.feature_list)}"
        model = RandomForestRegressor(n_estimators=1000, random_state=42)
        # model = RandomForestRegressor(n_estimators=10, random_state=42)
        model.fit(X, y)
        self.model = model
        self.trained = True

    def predict(self, X):
        assert self.trained, "model has not been trained"
        y_hat = self.model.predict(X)
        return y_hat

    def submit(self, test_df, test_Id):
        preds = self.predict(test_df)
        # for row in test_df.iterrows():
        #     print(row[1])
        #     preds = predictor.predict(row[1].values.reshape(1, -1))
        #     print(preds)
            
        submission = pd.DataFrame({"Id": test_Id.values, "SalePrice": preds})
        output_file = f"./output/submission_{self.name}.csv"

        submission.to_csv(output_file, index=False)
        log.info("submission saved to {}".format(output_file))

    def show_importance(self):
        assert self.trained, "model has not been trained"
        importances = list(self.model.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(self.feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=False)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[-10:]]

    def export(self):
        tree = self.model.estimators_[5]
        export_graphviz(tree, out_file='./output/tree.dot', feature_names=self.feature_list, rounded=True, precision=1, max_depth=3,  impurity=False,  proportion=True)
        (graph, ) = pydot.graph_from_dot_file('./output/tree.dot')
        graph.write_png('./output/tree.png')

def best_search():
    param_dist = {'n_estimators': randint(50, 2000),
              'max_depth': randint(1, 20)}

    train_X, test_X, train_y, test_y, feature_list = load_data()
    rf = RandomForestRegressor()
    rand_search = RandomizedSearchCV(rf, param_distributions=param_dist, n_iter=5, cv=5)
    rand_search.fit(train_X, train_y)
    best_rf = rand_search.best_estimator_
    print('Best hyperparameters:',  rand_search.best_params_)

    train_y_hat = best_rf.predict(train_X)
    train_rmse = rmse(train_y_hat, train_y)

    test_y_hat = best_rf.predict(test_X)
    test_rmse = rmse(test_y_hat, test_y)
    
    baseline = rmse(np.mean(train_y), train_y)
    print(f'baseline:{baseline:.1%}, train_rmse:{train_rmse:.1%}, test_rmse:{test_rmse:.1%}')





