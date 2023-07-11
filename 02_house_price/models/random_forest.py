import os
import sys
import pandas as pd
import numpy as np
import torch
from torch import nn
from torch.utils import data

from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn import metrics 
from scipy.stats import randint
from sklearn.tree import export_graphviz
import pydot

os.chdir(sys.path[0])



class RandomForestModel:

    def __init__(self):
        self.name = "random_forest"
        self.model = None

    def __call__(self, X, y):
        return self.train_model(X, y)

    def train_model(self, X, y):
        model = RandomForestRegressor(n_estimators=1000, random_state=42)
        model.fit(X, y)
        self.model = model
        return model

    def predict(self, X):
        assert self.model is not None, "model has not been trained"
        y_hat = self.model.predict(X)
        return y_hat

    def show_importance(self, feature_list):
        assert self.model is not None, "model has not been trained"
        importances = list(model.feature_importances_)
        feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
        feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=False)
        [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances[-10:]]

    def export(self, feature_list):
        tree = self.model.estimators_[5]
        export_graphviz(tree, out_file='./output/tree.dot', feature_names=feature_list, rounded=True, precision=1, max_depth=3,  impurity=False,  proportion=True)
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





