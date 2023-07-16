from logger import init_logger

train_input = "./data/kaggle_house_price/kaggle_house_pred_train.csv"
test_input = "./data/kaggle_house_price/kaggle_house_pred_test.csv"

metric = "log_rmse"  # [mape, rmse, log_rmse]
k_fold = 4

debug = True
log = init_logger("DEBUG" if debug else "INFO")

model_name = "random_forest"
model_name = "mlp"
model_name = "xgboost"