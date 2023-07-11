import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

os.chdir(sys.path[0])


class XGBModel:

    def __init__(self):
        self.name = "xgboost"
        self.model = None

    def __call__(self, X, y):
        return self.train_model(X, y)

    def train_model(self, X, y):
        # 追求最大精度
        params = {
            'learning_rate': 0.003,
            'n_estimators': 5000,
            'max_depth': 6,
            'min_child_weight': 3,
            'seed': 0,
            'subsample': 0.6,
            'colsample_bytree': 0.7,
            'gamma': 0,
            'reg_alpha': 0.2,
            'reg_lambda': 1
        }

        # 弱化过拟合
        # params = {'learning_rate': 0.03, 'n_estimators': 600, 'max_depth': 4, 'min_child_weight': 1, 'seed': 0,
        #                 'subsample': 0.6, 'colsample_bytree': 0.7, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 100}
        model = xgb.XGBRegressor(**params)
        model.fit(X, y)
        self.model = model
        return model

    def predict(self, X):
        assert self.model is not None, "model has not been trained"
        y_hat = self.model.predict(X)
        return y_hat

    def show_importance(self, feature_list):
        assert self.model is not None, "model has not been trained"
        feature_importance = pd.DataFrame(
            self.model.feature_importances_.reshape(-1, 1),
            index=feature_list).reset_index()
        feature_importance.columns = ["feature", "importance"]
        feature_importance.sort_values(by="importance", inplace=True)
        print(feature_importance.tail(10))


def run_grid_search(X, y):
    # cv_params = {'n_estimators': [400, 500, 600, 700, 800],'max_depth': [3,4,5,6,7],'learning_rate': [0.1, 0.05, 0.03, 0.2, 0.3]}
    # cv_params = {'learning_rate': [0.1, 0.05, 0.03, 0.2, 0.3],'max_depth': [3,4,5,6,7]}
    # cv_params = {'min_child_weight': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]}
    # cv_params = {'subsample':[i/10.0 for i in range(6,10)], 'colsample_bytree':[i/10.0 for i in range(6,10)]}
    cv_params = {
        'gamma': [i / 10.0 for i in range(0, 5)],
        'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
        'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]
    }
    cv_params = {'learning_rate': [i / 100.0 for i in range(1, 10)]}
    other_params = {
        'learning_rate': 0.03,
        'n_estimators': 600,
        'max_depth': 4,
        'min_child_weight': 1,
        'seed': 0,
        'subsample': 0.6,
        'colsample_bytree': 0.7,
        'gamma': 0,
        'reg_alpha': 0.1,
        'reg_lambda': 100
    }
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model,
                                 param_grid=cv_params,
                                 scoring='r2',
                                 cv=5,
                                 verbose=1,
                                 n_jobs=4)
    optimized_GBM.fit(X, y)
    evalute_result = optimized_GBM.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    regressor = optimized_GBM.best_estimator_
    return regressor, evalute_result
