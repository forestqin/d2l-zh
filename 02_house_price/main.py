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


def train_model(model_name, X_train, y_train, X_valid, y_valid, feature_list):
    # log.info(f"{model_name}: train start")
    if model_name == "xgboost":
        predictor = xgboost.XGBModel(feature_list)
        predictor.train(X_train, y_train)
        if config.debug:
            predictor.show_importance()
    elif model_name == "random_forest":
        predictor = random_forest.RandomForestModel(feature_list)
        predictor.train(X_train, y_train)
        if config.debug:
            predictor.show_importance()
    elif model_name == "mlp":
        model = mlp.MLPModel(feature_list)
        predictor = model.train(X_train, y_train, X_valid, y_valid)
    else:
        raise(Exception("Invalid model name"))
    return predictor

def main():
    log.info(f"Model: {config.model_name}")
    train_df, train_label, test_df, test_Id, feature_list = get_feature_df(config.train_input, config.test_input)
    feature_list = train_df.columns

    train_l_sum, valid_l_sum = 0, 0
    k = config.k_fold
    for i in range(k):
        X_train, y_train, X_valid, y_valid = get_k_fold_data(k, i, train_df, train_label)
        t = Timer()
        predictor = train_model(config.model_name, X_train, y_train, X_valid, y_valid, feature_list)
        spend_time = t.stop()
        train_score = calc_score(predictor, X_train, y_train)
        valid_score = calc_score(predictor, X_valid, y_valid)
        log.info(f"{config.metric} - {i+1}/{k}: {train_score=:.1f}, {valid_score=:.1f}, {spend_time=:.1f}s")

        train_l_sum += train_score
        valid_l_sum += valid_score
        break
        
    train_score = train_l_sum / (i+1)
    valid_score = valid_l_sum / (i+1)
    log.info(f"{config.metric} - {k}_fold average: {train_score=:.1f}, {valid_score=:.1f}")

    predictor.submit(test_df, test_Id)


if __name__ == "__main__":
    main()
