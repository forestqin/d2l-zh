import os
import sys
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error

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


def predict(predictor):
    test_data2 = pd.read_csv("./data/kaggle_house_price/kaggle_house_pred_test_processed.csv", sep="\t")
    preds = predictor.predict(test_data2.drop(columns=["Id"]))
    submission = pd.DataFrame({"Id":test_data2.Id, "SalePrice":preds})
    print(submission.head())
    submission.to_csv("./output/house_price_submission_xgboost.csv", index=False)
    print("predict finished")


def run_xgboost():
    train_X, test_X, train_y, test_y, feature_list = load_data()

    # 追求最大精度
    params = {'learning_rate': 0.003, 'n_estimators': 5000, 'max_depth': 6, 'min_child_weight': 3, 'seed': 0,
                    'subsample': 0.6, 'colsample_bytree': 0.7, 'gamma': 0, 'reg_alpha': 0.2, 'reg_lambda': 1}
    
    # 弱化过拟合
    # params = {'learning_rate': 0.03, 'n_estimators': 600, 'max_depth': 4, 'min_child_weight': 1, 'seed': 0,
    #                 'subsample': 0.6, 'colsample_bytree': 0.7, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 100}
    regressor = xgb.XGBRegressor(**params)

    regressor.fit(train_X, train_y)
    feature_importance = pd.DataFrame(regressor.feature_importances_.reshape(-1, 1), index=feature_list).reset_index()
    feature_importance.columns = ["feature", "importance"]
    feature_importance.sort_values(by="importance", inplace=True)
    print(feature_importance.tail(10))
    
    train_y_pred = regressor.predict(train_X)
    train_rmse_value = rmse(train_y, train_y_pred)

    test_y_pred = regressor.predict(test_X)
    rmse_value = rmse(test_y, test_y_pred)
    print(f"{train_rmse_value=:.2%}, {rmse_value=:.2%}")

    predict(regressor)


def run_grid_search():
    train_X, test_X, train_y, test_y, feature_list = load_data()
    # cv_params = {'n_estimators': [400, 500, 600, 700, 800],'max_depth': [3,4,5,6,7],'learning_rate': [0.1, 0.05, 0.03, 0.2, 0.3]}
    # cv_params = {'learning_rate': [0.1, 0.05, 0.03, 0.2, 0.3],'max_depth': [3,4,5,6,7]}
    # cv_params = {'min_child_weight': [0.5, 1, 1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6]}
    # cv_params = {'subsample':[i/10.0 for i in range(6,10)], 'colsample_bytree':[i/10.0 for i in range(6,10)]}
    cv_params = {'gamma':[i/10.0 for i in range(0,5)], 'reg_alpha':[1e-5, 1e-2, 0.1, 1, 100], 'reg_lambda':[1e-5, 1e-2, 0.1, 1, 100]}
    cv_params = {'learning_rate': [i/100.0 for i in range(1,10)]}
    other_params = {'learning_rate': 0.03, 'n_estimators': 600, 'max_depth': 4, 'min_child_weight': 1, 'seed': 0,
                    'subsample': 0.6, 'colsample_bytree': 0.7, 'gamma': 0, 'reg_alpha': 0.1, 'reg_lambda': 100}
    model = xgb.XGBRegressor(**other_params)
    optimized_GBM = GridSearchCV(estimator=model, param_grid=cv_params, scoring='r2', cv=5, verbose=1, n_jobs=4)
    optimized_GBM.fit(train_X, train_y)
    evalute_result = optimized_GBM.cv_results_
    # print('每轮迭代运行结果:{0}'.format(evalute_result))
    print('参数的最佳取值：{0}'.format(optimized_GBM.best_params_))
    print('最佳模型得分:{0}'.format(optimized_GBM.best_score_))

    regressor = optimized_GBM.best_estimator_
    feature_importance = pd.DataFrame(regressor.feature_importances_.reshape(-1, 1), index=feature_list).reset_index()
    feature_importance.columns = ["feature", "importance"]
    feature_importance.sort_values(by="importance", inplace=True)
    # print(feature_importance.tail(10))
    
    train_y_pred = regressor.predict(train_X)
    train_rmse_value = rmse(train_y, train_y_pred)

    test_y_pred = regressor.predict(test_X)
    rmse_value = rmse(test_y, test_y_pred)
    print(f"{train_rmse_value=:.2%}, {rmse_value=:.2%}")



if __name__ == "__main__":
    run_xgboost()
    # run_grid_search()
