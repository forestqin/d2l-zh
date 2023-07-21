import os
import sys
import pandas as pd
import numpy as np

import optuna
from xgboost import XGBRegressor
from preprocess_v2 import get_feature_df
import config
from sklearn.model_selection import KFold, cross_val_score

def score_dataset(X, y, model):
    # Label encoding for categoricals
    #
    # Label encoding is good for XGBoost and RandomForest, but one-hot
    # would be better for models like Lasso or Ridge. The `cat.codes`
    # attribute holds the category levels.
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_mean_squared_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

def objective(trial):
    xgb_params = dict(
        max_depth=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 1000, 8000),
        min_child_weight=trial.suggest_int("min_child_weight", 1, 10),
        colsample_bytree=trial.suggest_float("colsample_bytree", 0.2, 1.0),
        subsample=trial.suggest_float("subsample", 0.2, 1.0),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-4, 1e2, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-4, 1e2, log=True),
    )
    xgb = XGBRegressor(**xgb_params)
    return score_dataset(train_X, train_y, xgb)

if __name__ == "__main__":
    train_X, train_y, test_X, test_Id, feature_list = get_feature_df(config.train_input, config.test_input)
    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=20)
    xgb_params = study.best_params
    print(f"best xgb_params: {xgb_params}")

    xgb = XGBRegressor(**xgb_params)
    score = score_dataset(train_X, train_y, xgb)
    print(f"optuna best score: {score:.5f} RMSLE")

    # # Step 5 - Train Model and Create Submissions
    # X_train, X_test = create_features(df_train, df_test)
    # y_train = df_train.loc[:, "SalePrice"]

    # xgb = XGBRegressor(**xgb_params)
    # # XGB minimizes MSE, but competition loss is RMSLE
    # # So, we need to log-transform y to train and exp-transform the predictions
    # xgb.fit(X_train, np.log(y))
    # predictions = np.exp(xgb.predict(X_test))

    # output = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})
    # output.to_csv('output/submission_xgb_optuna.csv', index=False)
    # print("Your submission was successfully saved!")