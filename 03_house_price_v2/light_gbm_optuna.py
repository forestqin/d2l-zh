import os
import sys
import pandas as pd
import numpy as np

import optuna
from sklearn.model_selection import KFold, cross_val_score, train_test_split

from data_processing import get_feature_df_cached
import lightgbm as lgb
from sklearn.metrics import mean_squared_error
from lightgbm import LGBMRegressor

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

def objective_old(trial):
    # lgb.LGBMRegressor(objective='regression', num_leaves=5,
    #                           learning_rate=0.05, n_estimators=4000,
    #                           max_bin = 55, bagging_fraction = 0.8,
    #                           bagging_freq = 5, feature_fraction = 0.2,
    #                           feature_fraction_seed=9, bagging_seed=9,
    #                           min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)

    param = dict(objective='regression', 
        feature_fraction_seed=9, 
        bagging_seed=9,
        bagging_fraction=0.8,
        feature_fraction = 0.2,
        num_leaves=trial.suggest_int("max_depth", 2, 10),
        learning_rate=trial.suggest_float("learning_rate", 1e-4, 1e-1, log=True),
        n_estimators=trial.suggest_int("n_estimators", 1000, 20000),
        max_bin=trial.suggest_int("max_bin", 10, 100),
        bagging_freq=trial.suggest_int("bagging_freq", 3, 10),
        min_data_in_leaf=trial.suggest_int("min_data_in_leaf", 3, 10),
        min_sum_hessian_in_leaf=trial.suggest_int("min_sum_hessian_in_leaf", 5, 20)
    )
    
    lgb_model = lgb.LGBMRegressor(**param)
    return score_dataset(X_train, y_train, lgb_model)

X_train, y_train, X_test, Id_test = get_feature_df_cached()
X_test = X_test.fillna(0)

def objective(trial, data=X_train, target=y_train):
    train_x, test_x, train_y, test_y = train_test_split(data, target, test_size=0.2,random_state=42)
    param = {
        'metric': 'rmse', 
        'random_state': 48,
        'n_estimators': 20000,
        'reg_alpha': trial.suggest_loguniform('reg_alpha', 1e-3, 10.0),
        'reg_lambda': trial.suggest_loguniform('reg_lambda', 1e-3, 10.0),
        'colsample_bytree': trial.suggest_categorical('colsample_bytree', [0.3,0.4,0.5,0.6,0.7,0.8,0.9, 1.0]),
        'subsample': trial.suggest_categorical('subsample', [0.4,0.5,0.6,0.7,0.8,1.0]),
        'learning_rate': trial.suggest_categorical('learning_rate', [0.006,0.008,0.01,0.014,0.017,0.02]),
        'max_depth': trial.suggest_categorical('max_depth', [10,20,100]),
        'num_leaves' : trial.suggest_int('num_leaves', 1, 1000),
        'min_child_samples': trial.suggest_int('min_child_samples', 1, 300),
        'cat_smooth' : trial.suggest_int('min_data_per_groups', 1, 100)
    }
    model = LGBMRegressor(**param)  
    
    model.fit(train_x,train_y,eval_set=[(test_x,test_y)],early_stopping_rounds=100,verbose=False)
    
    preds = model.predict(test_x)
    
    rmse = mean_squared_error(test_y, preds,squared=False)
    
    return rmse

if __name__ == "__main__":
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=50)
    print('Number of finished trials:', len(study.trials))
    best_params = study.best_trial.params
    print('Best trial:', best_params)

    lgbm_model = lgb.LGBMRegressor(**best_params)
    score = score_dataset(X_train, y_train, lgbm_model)
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