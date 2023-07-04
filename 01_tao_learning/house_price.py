import os
import sys
import pandas as pd
import numpy as np

os.chdir(sys.path[0])

if __name__ == "__main__":
    df_train = pd.read_csv(
        "./data/kaggle_house_price/kaggle_house_pred_train_processed.csv",
        sep="\t")
    print(df_train.head())
