from logger import init_logger

train_input = "./data/kaggle_house_price/kaggle_house_pred_train.csv"
test_input = "./data/kaggle_house_price/kaggle_house_pred_test.csv"

metric = "mape"  # [mape, rmse]
k_fold = 4

debug = False
log = init_logger("d2l", "DEBUG" if debug else "INFO")

model_name = "random_forest"
model_name = "xgboost"
model_name = "mlp"