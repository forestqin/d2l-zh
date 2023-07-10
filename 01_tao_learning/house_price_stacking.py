import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
import pickle

from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
from sklearn.linear_model import LogisticRegression
from mlxtend.regressor import StackingRegressor
from sklearn.ensemble import RandomForestRegressor, ExtraTreesRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge
from sklearn.svm import SVR

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

def predict(regressor, train_X, train_y, test_X, test_y):
    train_y_pred = regressor.predict(train_X)
    train_rmse_value = rmse(train_y, train_y_pred)

    test_y_pred = regressor.predict(test_X)
    rmse_value = rmse(test_y, test_y_pred)
    print(f"{train_rmse_value=:.2%}, {rmse_value=:.2%}")


def run_stacking():
    train_X, test_X, train_y, test_y, feature_list = load_data()

    # 追求最大精度
    params = {'learning_rate': 0.003, 'n_estimators': 3000, 'max_depth': 6, 'min_child_weight': 3, 'seed': 0,
                    'subsample': 0.6, 'colsample_bytree': 0.7, 'gamma': 0, 'reg_alpha': 0.2, 'reg_lambda': 1}
    
    # 弱化过拟合
    # params = {'learning_rate': 0.03, 'n_estimators': 600, 'max_depth': 4, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.6, 'colsample_bytree': 0.7, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 100}
    xgb_reg = xgb.XGBRegressor(**params)
    rf = RandomForestRegressor(n_estimators=1000, random_state=42)
    
    etc = ExtraTreesRegressor(n_jobs=-1, n_estimators=5, criterion="squared_error")
    
    logr = LogisticRegression(n_jobs=-1, C=8)  # meta classifier
    svr_lin = SVR(kernel='linear')
    ridge = Ridge(random_state=1)
    svr_rbf = SVR(kernel='rbf')

    lr = LinearRegression()

    regressor = StackingRegressor(regressors=[xgb_reg, rf], 
                                  meta_regressor=lr)
    
    regressor.fit(train_X, train_y)
    print("training finished")
    predict(regressor, train_X, train_y, test_X, test_y)

    # save model for later predicting
    with open(r'./output/stacking.pkl', 'wb') as f:
        pickle.dump(regressor, f)
    print("model dumped")

    # with open('../data/training_df.pkl', 'rb') as f:
    #     df = pickle.load(f)

    





if __name__ == "__main__":
    run_stacking()
