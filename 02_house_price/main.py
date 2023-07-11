import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

import config
import preprocess
import utils
from models import xgboost, random_forest, mlp

os.chdir(sys.path[0])

def train_model(model_name, train_X, train_y, val_X, val_y):
    if model_name == "xgboost":
        model = xgboost.XGBModel()
        predictor = model(train_X, train_y)
    elif model_name == "random_forest":
        model = random_forest.RandomForestModel()
        predictor = model(train_X, train_y)
    elif model_name == "mlp":
        model = mlp.MLPModel()
        predictor = model(train_X, train_y, val_X, val_y)
    else:
        raise(Exception("Invalid model name"))
    
    return predictor


model_name = "xgboost"
model_name = "random_forest"
model_name = "mlp"

if __name__ == "__main__":
    train_df = preprocess.gen_feature_dataframe(config.train_input, is_train=True)
    test_df = preprocess.gen_feature_dataframe(config.test_input, is_train=False)
    train_X, val_X, train_y, val_y, feature_list = utils.gen_train_test_dataset(train_df)

    t = utils.Timer()
    print(f"train start")
    predictor = train_model(model_name, train_X, train_y, val_X, val_y)
    print(f"train spend in {t.stop():.1f}s")

    train_mre = utils.calc_mre(predictor, train_X, train_y)
    validate_mre = utils.calc_mre(predictor, val_X, val_y)
    print(f"{model_name}: {train_mre=:.2%}, {validate_mre=:.2%}")
    submission = utils.gen_submission(predictor, test_df, model_name)