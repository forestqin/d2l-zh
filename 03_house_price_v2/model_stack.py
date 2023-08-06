# Basic
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Visualization
import seaborn as sns
import sklearn_pandas

# Encoding
from sklearn.base import BaseEstimator, TransformerMixin, ClassifierMixin, clone
from sklearn.impute import SimpleImputer, MissingIndicator
from sklearn.preprocessing import FunctionTransformer, LabelEncoder, Normalizer, StandardScaler, OneHotEncoder

# Models
from sklearn.linear_model import Ridge, RidgeCV, ElasticNet, LassoCV, LassoLarsCV
from sklearn.model_selection import cross_val_score, KFold
from sklearn.pipeline import make_pipeline
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import xgboost as xgb
from sklearn.linear_model import ElasticNet, Lasso, ElasticNetCV,LinearRegression
from sklearn.ensemble import  GradientBoostingRegressor
import lightgbm as lgb

# metrics
from sklearn.metrics import mean_squared_error,accuracy_score

# Warning
import warnings
warnings.filterwarnings('ignore')

# qintao
from data_processing import get_feature_df, get_feature_df_cached

def rmse(model):
    n_folds=5
    kfold = KFold(n_folds, random_state=42, shuffle=True).get_n_splits(X_train)
    rmse_score = np.sqrt(-cross_val_score(model, X_train, y_train, scoring = "neg_mean_squared_error", cv = kfold, verbose = -1, n_jobs=-1))
    return(np.mean(rmse_score))


def show(y_train, y_train_pred):
    plt.figure(figsize=(14,6))
    plt.scatter(y_train, y_train_pred)
    plt.xlabel("Prices")
    plt.ylabel("Predicted prices")
    plt.title("Prices vs. Predicted Prices")
    plt.show()

    plt.figure(figsize=(14,5))
    plt.scatter(y_train_pred,y_train_pred - y_train)
    plt.title("Residual Plot")
    plt.xlabel("Predicted values")
    plt.ylabel("Residuals")
    plt.tight_layout()
    plt.show()


def run_linear_model(X_train, y_train):
    lr_model = make_pipeline(RobustScaler(), LinearRegression())
    lr_model.fit(X_train,y_train)
    y_train_pred = lr_model.predict(X_train)
    mse_train = round(mean_squared_error(y_train_pred, y_train), 5)
    rmse_lr = round(rmse(lr_model),5)
    print('MSE for Linear Regression is :',mse_train)
    print('RMSE for Linear Regression is :',rmse_lr)
    return lr_model, y_train_pred

def run_lasso_model(X_train, y_train):
    ls_model = make_pipeline(RobustScaler(),LassoCV(alphas=[0.0005],random_state=0,cv=10))
    ls_model.fit(X_train, y_train)
    y_train_pred = ls_model.predict(X_train)
    mse_train = round(mean_squared_error(y_train_pred, y_train), 5)
    rmse_lasso = round(rmse(ls_model),5)
    print('MSE for Lasso Regression is :',mse_train)
    print('RMSE for Lasso Regression is :',rmse_lasso)
    return ls_model, y_train_pred

def run_gbr_model(X_train, y_train):
    gbr = GradientBoostingRegressor(random_state=0)
    param_grid = {'n_estimators': [3400],
                'max_features': [13],
                'max_depth': [5],
                'learning_rate': [0.01],
                'subsample': [0.8],
                'random_state' : [5]}
    gb_model = GridSearchCV(estimator=gbr, param_grid=param_grid, n_jobs=1, cv=5)
    gb_model.fit(X_train, y_train)
    #gb_model.best_params3
    y_train_pred = gb_model.predict(X_train)
    mse_train = round(mean_squared_error(y_train_pred, y_train),5)
    rmse_gb = round(rmse(gb_model),5)
    print('MSE for GB Regression is :',mse_train)
    print('RMSE for GB Regression is :',rmse_gb)
    return gb_model, y_train_pred


def run_xgb_model(X_train, y_train):
    xgbreg = xgb.XGBRegressor(seed=0)
    param_grid2 = {'n_estimators': [2500], 
                'learning_rate': [0.03],
                'max_depth': [3],
                'subsample': [0.8],
                'colsample_bytree': [0.45]}
        
    xgb_model = GridSearchCV(estimator=xgbreg, param_grid=param_grid2, n_jobs=1, cv=10)
    xgb_model.fit(X_train, y_train)
    y_train_pred = xgb_model.predict(X_train)
    mse_train = round(mean_squared_error(y_train_pred, y_train), 5)
    rmse_xgb = round(rmse(xgb_model),5)
    print('MSE for XGB is :',mse_train)
    print('RMSE for XGB is :',rmse_xgb)
    return xgb_model, y_train_pred

def run_en_model(X_train, y_train):
    en_model = ElasticNetCV(alphas = [0.0005], 
                        l1_ratio = [.9], 
                        random_state = 0,
                        cv=10)
    en_model.fit(X_train,y_train)   
    y_train_pred = en_model.predict(X_train)
    mse_train = round(mean_squared_error(y_train_pred,y_train),5)
    rmse_en = round(rmse(en_model),5)
    print('MSE for Elastic Net is :',mse_train)
    print('RMSE for Elastic Net is :',rmse_en)
    return en_model, y_train_pred

def run_light_gbm_model(X_train, y_train):
    lgb_model = lgb.LGBMRegressor(objective='regression', num_leaves=5,
                              learning_rate=0.05, n_estimators=4000,
                              max_bin = 55, bagging_fraction = 0.8,
                              bagging_freq = 5, feature_fraction = 0.2,
                              feature_fraction_seed=9, bagging_seed=9,
                              min_data_in_leaf =6, min_sum_hessian_in_leaf = 11)
    lgb_model.fit(X_train,y_train)
    y_train_pred = lgb_model.predict(X_train)
    mse_train = round(mean_squared_error(y_train_pred,y_train),5)
    rmse_lgb = round(np.sqrt(mean_squared_error(y_train_pred,y_train)),5)
    print('MSE for Light GBM is :',mse_train)
    print('RMSE for Light GBM is :',rmse_lgb)
    return lgb_model, y_train_pred

def run_stacking_model(X_train, y_train):
    from mlxtend.regressor import StackingCVRegressor
    lasso_model = make_pipeline(RobustScaler(), 
                            LassoCV(max_iter= 10000000, alphas = [0.0005],random_state = 42, cv=5))

    elasticnet_model = make_pipeline(RobustScaler(),
                                    ElasticNetCV(max_iter=10000000, alphas=[0.0005], cv=5, l1_ratio=0.9))

    lgbm_model = make_pipeline(RobustScaler(),
                            lgb.LGBMRegressor(objective='regression',num_leaves=5,
                                                learning_rate=0.05, n_estimators=4000,
                                                max_bin = 55, bagging_fraction = 0.8,
                                                bagging_freq = 5, feature_fraction = 0.23,
                                                feature_fraction_seed = 9, bagging_seed=9,
                                                min_data_in_leaf = 6, 
                                                min_sum_hessian_in_leaf = 11))

    xgboost_model = make_pipeline(RobustScaler(),
                                xgb.XGBRegressor(learning_rate = 0.01, n_estimators=3400,
                                                max_depth=3,min_child_weight=0 ,
                                                gamma=0, subsample=0.7,colsample_bytree=0.7,
                                                objective= 'reg:linear',nthread=4,
                                                scale_pos_weight=1,seed=27, reg_alpha=0.00006))

    stack_regressor = StackingCVRegressor(regressors=(lasso_model, elasticnet_model, xgboost_model, lgbm_model),
                                        meta_regressor=xgboost_model, use_features_in_secondary=True)
    stack_model = stack_regressor.fit(np.array(X_train),  np.array(y_train))
    y_train_pred = stack_model.predict(X_train)
    mse_train = round(mean_squared_error(y_train_pred,y_train),5)
    rmse_lgb = round(np.sqrt(mean_squared_error(y_train_pred,y_train)),5)
    print('MSE for Stacking Model is :',mse_train)
    print('RMSE for Stacking Model is :',rmse_lgb)
    return stack_model, y_train_pred

def run_weighted_stacking_model(X_train, y_train, stacking_model, lgb_model, lasso_model, en_model, xgb_model, gbr_model):
    stack_gen_pred = stacking_model.predict(X_train)
    lgbm_pred = lgb_model.predict(X_train)
    lasso_pred = lasso_model.predict(X_train)
    en_pred = en_model.predict(X_train)
    xgb_pred = xgb_model.predict(X_train)
    gb_pred = gbr_model.predict(X_train)
    y_train_pred = ((0.1*xgb_pred) + (0.075*gb_pred) + (0.4*lgbm_pred) + (0.4*stack_gen_pred) +(0.025*en_pred) ) 
    mse_train = round(mean_squared_error(y_train_pred, y_train), 5)
    rmse_lgb = round(np.sqrt(mean_squared_error(y_train_pred, y_train)), 5)
    print('MSE for Weighted Stacking Model is :',mse_train)
    print('RMSE for Weighted Stacking Model is :',rmse_lgb)
    return None, y_train_pred


def run_stacking_predict(X_test, stacking_model, lgb_model, lasso_model, en_model, xgb_model, gbr_model):
    stack_gen_pred = stacking_model.predict(X_test)
    lgbm_pred = lgb_model.predict(X_test)
    lasso_pred = lasso_model.predict(X_test)
    en_pred = en_model.predict(X_test)
    xgb_pred = xgb_model.predict(X_test)
    gb_pred = gbr_model.predict(X_test)
    stack_preds = ((0.1*xgb_pred) + (0.075*gb_pred) + (0.4*lgbm_pred) + (0.4*stack_gen_pred) +(0.025*en_pred) ) 
    final_pred = np.expm1(stack_preds)

    return final_pred

def run_lgbm():
    X_train, y_train, X_test, Id_test = get_feature_df_cached()
    X_test = X_test.fillna(0)
    lgb_model, y_train_pred = run_light_gbm_model(X_train, y_train)
    show(y_train, y_train_pred)
    lgbm_pred = lgb_model.predict(X_test)
    final_pred = np.expm1(lgbm_pred)
    output = pd.DataFrame({'Id': Id_test, 'SalePrice': final_pred})
    output.to_csv('output/submission_lgbm.csv', index=False)
    print("Your submission was successfully saved!")

def main():
    # train_X, train_y, test_X, test_Id = get_feature_df()
    X_train, y_train, X_test, Id_test = get_feature_df_cached()
    X_test = X_test.fillna(0)

    lm_model, y_train_pred = run_linear_model(X_train, y_train)
    lasso_model, y_train_pred = run_lasso_model(X_train, y_train)
    gbr_model, y_train_pred = run_gbr_model(X_train, y_train)
    xgb_model, y_train_pred = run_xgb_model(X_train, y_train)
    en_model, y_train_pred = run_en_model(X_train, y_train)
    lgb_model, y_train_pred = run_light_gbm_model(X_train, y_train)
    stacking_model, y_train_pred = run_stacking_model(X_train, y_train)

    _, y_train_pred = run_weighted_stacking_model(X_train, y_train, stacking_model, lgb_model, lasso_model, en_model, xgb_model, gbr_model)
    show(y_train, y_train_pred)

    final_pred = run_stacking_predict(X_test, stacking_model, lgb_model, lasso_model, en_model, xgb_model, gbr_model)
    output = pd.DataFrame({'Id': Id_test, 'SalePrice': final_pred})
    output.to_csv('output/submission_stacking.csv', index=False)
    print("Your submission was successfully saved!")

if __name__ == "__main__":
    # run_lgbm()
    main()