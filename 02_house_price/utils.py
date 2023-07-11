import pandas as pd
import numpy as np
import time
import datetime
from sklearn.model_selection import train_test_split
import config


def gen_train_test_dataset(df):
    feature_list = df.columns[:-1]
    train_features = df.iloc[:, :-1].values
    train_labels = df.iloc[:, -1].values
    train_X, test_X, train_y, test_y = train_test_split(train_features,
                                                        train_labels,
                                                        test_size=config.test_size,
                                                        random_state=42)
    print('Training Features Shape:', train_X.shape)
    print('Training Labels Shape:', train_y.shape)
    print('Testing Features Shape:', test_X.shape)
    print('Testing Labels Shape:', test_y.shape)
    return train_X, test_X, train_y, test_y, feature_list


def calc_mre(regressor, X, y):
    y_hat = regressor.predict(X)
    mre = np.mean(abs(y_hat / y - 1))
    return mre


def gen_submission(predictor, test_df, suffix):
    preds = predictor.predict(test_df.drop(columns=["Id"]))
    submission = pd.DataFrame({"Id": test_df.Id, config.target: preds})
    # dateArray = datetime.datetime.fromtimestamp(time.time())
    # output_file = "./output/house_price_submission_{}_{}.csv".format(
    #     suffix, dateArray.strftime("%Y%m%d_%H%M%S"))
    output_file = f"./output/submission_{suffix}.csv"

    submission.to_csv(output_file, index=False)
    print("submission saved to {}".format(output_file))
    return submission

class Timer:
    def __init__(self):
        self.times = []
        self.start()

    def start(self):
        self.tik = time.time()

    def stop(self):
        self.times.append(time.time() - self.tik)
        return self.times[-1]

    def avg(self):
        return sum(self.times) / len(self.times)

    def sum(self):
        return sum(self.times)

    def cumsum(self):
        return np.array(self.times).cumsum().tolist()