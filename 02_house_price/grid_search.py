import os
import sys
import pandas as pd
import numpy as np

import config
from config import log
from preprocess import get_feature_df
from utils import Timer, get_k_fold_data, calc_score
from models import xgboost, random_forest, mlp

os.chdir(sys.path[0])

def grid_search_xgboost():
    train_df, train_label, test_df, test_Id, feature_list = get_feature_df(config.train_input, config.test_input)
    X_train, y_train, X_valid, y_valid = get_k_fold_data(config.k_fold, 0, train_df, train_label)
    t = Timer()
    predictor = xgboost.run_grid_search(X_train, y_train)
    spend_time = t.stop()
    train_score = calc_score(predictor, X_train, y_train)
    valid_score = calc_score(predictor, X_valid, y_valid)
    log.info(f"{config.metric}: {train_score=:.3f}, {valid_score=:.3f}, {spend_time=:.1f}s")

if __name__ == "__main__":
    grid_search_xgboost()
