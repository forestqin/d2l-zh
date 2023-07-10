import pandas as pd
import numpy as np
import matplotlib.pyplot as mp
from autogluon.tabular import TabularDataset, TabularPredictor

df = pd.read_csv("./data/kaggle_house_price/kaggle_house_pred_train_processed.csv", sep="\t")
df.head()

train_data = df.sample(frac=0.75)  # 80% 的样本作为训练集
test_data = df.drop(train_data.index)  # 剩下的样本作为测试集

train_data.shape, test_data.shape
train_data = TabularDataset(df)

label = 'SalePrice'

predictor = TabularPredictor(label=label).fit(train_data)

predictor.leaderboard(test_data, silent=True)

predictor.evaluate(test_data, silent=True)

def rmse(y_hat, y):
    return np.mean(abs(y_hat / y - 1))

def predict(regressor, train_data, test_data):
    train_y_pred = regressor.predict(train_data)
    train_y = train_data.iloc[:, -1].values
    train_rmse_value = rmse(train_y, train_y_pred)

    test_y_pred = regressor.predict(test_data)
    test_y = test_data.iloc[:, -1].values
    rmse_value = rmse(test_y, test_y_pred)
    print(f"{train_rmse_value=:.2%}, {rmse_value=:.2%}")

predict(predictor, train_data, test_data)

test_data2 = pd.read_csv("./data/kaggle_house_price/kaggle_house_pred_test_processed.csv", sep="\t")
test_data2 = TabularDataset(test_data2)
test_data2.head()

train_data.shape, test_data.shape, test_data2.shape

set(test_data2.columns) - set(train_data.columns)

preds = predictor.predict(test_data2.drop(columns=["Id"]))


submission = pd.DataFrame({"Id":test_data2.Id, label:preds})
submission.head()

submission.to_csv("./output/house_price_submission_autogluen.csv", index=False)