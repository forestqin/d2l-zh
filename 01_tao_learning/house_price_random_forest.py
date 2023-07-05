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


def load_data():
    df_train = pd.read_csv(
        "./data/kaggle_house_price/kaggle_house_pred_train_processed.csv",
        sep="\t")
    feature_list = df_train.iloc[:, 1:].columns
    train_features = df_train.iloc[:, :-1].values
    train_labels = df_train.iloc[:, -1].values
    train_X, test_X, train_y, test_y = train_test_split(train_features,
                                                        train_labels,
                                                        test_size=0.25,
                                                        random_state=42)
    print('Training Features Shape:', train_X.shape)
    print('Training Labels Shape:', train_y.shape)
    print('Testing Features Shape:', test_X.shape)
    print('Testing Labels Shape:', test_y.shape)
    return train_X, test_X, train_y, test_y, feature_list

def rmse(y_hat, y):
    return np.mean(abs(y_hat / y - 1))

def run_one():
    train_X, test_X, train_y, test_y, feature_list = load_data()
    
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    rf.fit(train_X, train_y)

    train_y_hat = rf.predict(train_X)
    train_rmse = rmse(train_y_hat, train_y)

    test_y_hat = rf.predict(test_X)
    test_rmse = rmse(test_y_hat, test_y)
    
    baseline = rmse(np.mean(train_y), train_y)
    print(f'baseline:{baseline:.1%}, train_rmse:{train_rmse:.1%}, test_rmse:{test_rmse:.1%}')

    importances = list(rf.feature_importances_)
    feature_importances = [(feature, round(importance, 2)) for feature, importance in zip(feature_list, importances)]
    feature_importances = sorted(feature_importances, key=lambda x: x[1], reverse=False)
    [print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances]

    export(rf, feature_list)

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


def export(rf, feature_list):
    tree = rf.estimators_[5]
    export_graphviz(tree, out_file='./output/tree.dot', feature_names=feature_list, rounded=True, precision=1, max_depth=3,  impurity=False,  proportion=True)
    (graph, ) = pydot.graph_from_dot_file('./output/tree.dot')
    graph.write_png('./output/tree.png')


if __name__ == "__main__":
    run_one()
    # best_search()
