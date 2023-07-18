import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from config import log
# import optuna


os.chdir(sys.path[0])


class XGBModel:

    def __init__(self, feature_list):
        self.name = "xgboost"
        self.feature_list = feature_list
        self.model = None
        self.trained = False

    def train(self, X, y):
        assert X.shape[1] == len(self.feature_list), f"数据集{X.shape}不匹配特征数量{len(self.feature_list)}"
        # 追求最大精度
        params = {
            'learning_rate': 0.003,
            'n_estimators': 4000,
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
        self.trained = True

    def predict(self, X):
        assert self.trained, "model has not been trained"
        y_hat = self.model.predict(X)
        return y_hat


    def show_importance(self):
        assert self.trained, "model has not been trained"
        feature_importance = pd.DataFrame(
            self.model.feature_importances_.reshape(-1, 1),
            index=self.feature_list).reset_index()
        feature_importance.columns = ["feature", "importance"]
        feature_importance.sort_values(by="importance", inplace=True)
        print(feature_importance.tail(10))
        # return feature_importance

    def submit(self, test_df, test_Id):
        preds = self.predict(test_df)
        submission = pd.DataFrame({"Id": test_Id.values, "SalePrice": preds})
        output_file = f"./output/submission_{self.name}.csv"

        submission.to_csv(output_file, index=False)
        log.info("submission saved to {}".format(output_file))

def run_grid_search(X, y):
    # cv_params = {'n_estimators': [400, 500, 600, 700, 800],'max_depth': [3, 4, 5, 6, 7],'learning_rate': [0.1, 0.05, 0.03, 0.2, 0.3]}
    # cv_params = {'learning_rate': [0.1, 0.05, 0.03, 0.2, 0.3],'max_depth': [3,4,5,6,7]}
    # cv_params = {'min_child_weight': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]}
    # cv_params = {'subsample':[i/10.0 for i in range(6,10)], 'colsample_bytree':[i/10.0 for i in range(6,10)]}
    # cv_params = {
    #     'gamma': [i / 10.0 for i in range(0, 5)],
    #     'reg_alpha': [1e-5, 1e-2, 0.1, 1, 100],
    #     'reg_lambda': [1e-5, 1e-2, 0.1, 1, 100]
    # }
    # cv_params = {'learning_rate': [i / 1000.0 for i in range(1, 10)]}
    cv_params = {'n_estimators': [400, 500, 600, 700, 800]}
    params = {
            'learning_rate': 0.003,
            'n_estimators': 1000,
            'max_depth': 8,
            'min_child_weight': 4,
            'seed': 0,
            'subsample': 0.8,
            'colsample_bytree': 0.7,
            'gamma': 0,
            'reg_alpha': 0,
            'reg_lambda': 0
        }
    model = xgb.XGBRegressor(**params)
    optimized_GBM = GridSearchCV(estimator=model,
                                 param_grid=cv_params,
                                 scoring='r2',
                                 cv=4,
                                 verbose=1,
                                 n_jobs=20)
    optimized_GBM.fit(X, y)
    evalute_result = optimized_GBM.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    regressor = optimized_GBM.best_estimator_
    return regressor


def score_dataset(X, y, model=xgb.XGBRegressor()):
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
    return score_dataset(X_train, y_train, xgb)