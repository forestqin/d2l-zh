import os
import sys
import pandas as pd
import numpy as np
import config
from config import log
import utils
import torch
from sklearn import preprocessing


os.chdir(sys.path[0])

target = "SalePrice"

ignore_list = ["Id", "Utilities", "Condition2", "3SsnPorch", "PoolQC", 
    "MoSold", "YrSold", "MiscFeature"]

dummy_list = ["MSSubClass", "MSZoning", "LotFrontage", "Street", "Alley", 
    "LotShape", "LandContour", "LotConfig", "LandSlope", "Neighborhood", 
    "Condition1", "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", 
    "Exterior1st", "Exterior2nd", "MasVnrType", "ExterCond", "Foundation", 
    "BsmtCond", "BsmtExposure", "BsmtFinType1", "BsmtFinType2", "Heating", 
    "CentralAir", "Electrical", "BsmtHalfBath", "BedroomAbvGr", "KitchenAbvGr", 
    "Functional", "GarageType", "GarageFinish", "PavedDrive", "Fence", 
    "SaleType", "SaleCondition"
]

log_numeric_list = ["LotArea", "MasVnrArea", "BsmtFinSF1", "BsmtFinSF2", 
    "BsmtUnfSF", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "GrLivArea", 
    "GarageArea", "WoodDeckSF", "OpenPorchSF", "EnclosedPorch", "ScreenPorch"]

numeric_list = [
    "OverallQual", "OverallCond", "YearBuilt", "YearRemodAdd", 
    "BsmtFullBath", "FullBath", "HalfBath", "TotRmsAbvGrd", 
    "Fireplaces", "GarageYrBlt", "GarageCars"]

Grade = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
grade_dict = {
    "ExterQual": Grade,
    "BsmtQual": Grade,
    "HeatingQC": Grade,
    "KitchenQual": Grade,
    "FireplaceQu": Grade,
    "GarageQual": Grade,
    "GarageCond": Grade
}


def complex_process(df):
    df["MasVnrArea_01"] = df["MasVnrArea"].apply(lambda x: 0 if x == 0 else 1)
    df["BsmtFinSF1_01"] = df["BsmtFinSF1"].apply(lambda x: 0 if x == 0 else 1)
    df["BsmtUnfSF_01"] = df["BsmtUnfSF"].apply(lambda x: 0 if x == 0 else 1)
    df["TotalBsmtSF_01"] = df["TotalBsmtSF"].apply(lambda x: 0 if x == 0 else 1)
    df["2ndFlrSF_01"] = df["2ndFlrSF"].apply(lambda x: 0 if x == 0 else 1)
    df["LowQualFinSF_01"] = df["LowQualFinSF"].apply(lambda x: 0 if x == 0 else 1)
    df["WoodDeckSF_01"] = df["WoodDeckSF"].apply(lambda x: 0 if x == 0 else 1)
    df["OpenPorchSF_01"] = df["OpenPorchSF"].apply(lambda x: 0 if x == 0 else 1)
    df["EnclosedPorch_01"] = df["EnclosedPorch"].apply(lambda x: 0 if x == 0 else 1)
    df["ScreenPorch_01"] = df["ScreenPorch"].apply(lambda x: 0 if x == 0 else 1)
    df["PoolArea_01"] = df["PoolArea"].apply(lambda x: 0 if x == 0 else 1)
    df["MiscVal_01"] = df["MiscVal"].apply(lambda x: 0 if x == 0 else 1)
    # df["YearRemodAdd_Grade"] = df["YearRemodAdd"].apply(year_grade)
    df["GarageYrBlt"] = df["GarageYrBlt"].fillna(1950)

    dl = ["MasVnrArea_01", "BsmtFinSF1_01", "BsmtUnfSF_01", "TotalBsmtSF_01", "2ndFlrSF_01", 
        "LowQualFinSF_01", "WoodDeckSF_01", "OpenPorchSF_01", "EnclosedPorch_01", 
        "ScreenPorch_01", "PoolArea_01", "MiscVal_01"]
    dummy_list.extend(dl)

    lbl = preprocessing.LabelEncoder()
    new_df = pd.DataFrame()
    df_list = []
    for col in df.columns:
        if col in dummy_list:
            # dummy_df = pd.get_dummies(df[col].fillna("NA"), prefix=col)
            # df_list.append(dummy_df)
            new_df[col] = lbl.fit_transform(df[col])
        elif col in log_numeric_list:
            d = np.log(df[col].fillna(0).astype("float32")+1)
            d = (d - d.mean()) / d.std()
            new_df[col] = d.values  # 标准化
        elif col in numeric_list:
            d = df[col].fillna(0).astype("float32")
            d = (d - d.mean()) / d.std()
            new_df[col] = d.values  # 标准化
        elif col in grade_dict:
            dic = grade_dict[col]
            d = df[col].fillna("NA").map(dic)
            # new_df[col] = (d - d.mean()) / d.std()  # 标准化
        else:
            continue
    df_list.append(new_df)
    feature_df = pd.concat(df_list, axis=1)
    return feature_df


def simple_process(all_features):
    numeric_features = all_features.dtypes[all_features.dtypes != 'object'].index
    all_features[numeric_features] = all_features[numeric_features].apply(
        lambda x: (x - x.mean()) / (x.std()))
    all_features[numeric_features] = all_features[numeric_features].fillna(0)
    all_features = pd.get_dummies(all_features, dummy_na=True)
    log.debug(all_features.shape)
    return all_features


def get_feature_df(train_input, test_input):
    t = utils.Timer()
    train = pd.read_csv(train_input)
    test = pd.read_csv(test_input)
    train_data = train.drop(['Id', target], axis=1)
    test_data = test.drop(['Id'], axis=1)
    all_features = pd.concat((train_data.iloc[:, :], test_data.iloc[:, :]))
    
    # feature_df = simple_process(all_features)
    feature_df = complex_process(all_features)

    n_train = train.shape[0]
    train_df = feature_df[:n_train]
    train_label = train[target]
    # train_label = train[target].apply(lambda x: np.log(x+1))
    test_df  = feature_df[n_train:]
    test_Id = test['Id']
    feature_list = feature_df.columns.tolist()


    log.info(f'Training Shape: {train_df.shape}, {train_label.shape}')
    log.info(f'Testing  Shape: {test_df.shape}, {test_Id.shape}')
    log.info(f"process data in {t.stop():.1f}s")

    return train_df, train_label, test_df, test_Id, feature_list


if __name__ == "__main__":
    train_df, train_label, test_df, test_Id, feature_list = get_feature_df(config.train_input, config.test_input)
