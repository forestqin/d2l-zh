import os
import sys
import pandas as pd
import numpy as np
import config
from config import log
import utils
import torch


os.chdir(sys.path[0])

ignore_list = [
    "Id", "LotFrontage", "Street", "Alley", "Utilities", "Condition2",
    "YearBuilt", "RoofMatl", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF",
    "Electrical", "PoolArea", "PoolQC", "MiscVal", "MoSold", "YrSold",
    "MSSubClass", "MSZoning", "HouseStyle", "Exterior1st", "Exterior2nd",
    "Heating", "MiscFeature", "SaleType"
]

dummy_list = [
    "LotConfig", "Neighborhood", "Condition1", "BldgType", "RoofStyle",
    "MasVnrType", "Foundation", "CentralAir", "GarageType", "PavedDrive",
    "Fence", "SaleCondition"
]

numeric_list = [
    "LotArea", "OverallQual", "OverallCond", "YearRemodAdd", "MasVnrArea",
    "BsmtFinSF1", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
    "EnclosedPorch", "3SsnPorch", "ScreenPorch"
]

LotShape = {"NA": 4, "Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1}
LandContour = {"NA": 1, "Low": 1, "Bnk": 2, "HLS": 3, "Lvl": 4}
LandSlope = {"NA": 3, "Sev": 1, "Mod": 3, "Gtl": 5}
ExterQual = {"NA": 3, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
BsmtQual = {"NA": 3, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
BsmtExposure = {"NA": 3, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
BsmtFinType1 = {
    "NA": 3,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6
}
HeatingQC = {"NA": 3, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
KitchenQual = {"NA": 3, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
Functional = {
    "NA": 5, 
    "Sal": 1,
    "Sev": 2,
    "Maj2": 3,
    "Maj1": 4,
    "Mod": 5,
    "Min2": 6,
    "Min1": 7,
    "Typ": 8
}
FireplaceQu = {"NA": 3, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
GarageFinish = {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3}
GarageQual = {"NA": 3, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
grade_dict = {
    "LotShape": LotShape,
    "LandContour": LandContour,
    "LandSlope": LandSlope,
    "ExterQual": ExterQual,
    "ExterCond": ExterQual,
    "BsmtQual": BsmtQual,
    "BsmtCond": BsmtQual,
    "BsmtExposure": BsmtExposure,
    "BsmtFinType1": BsmtFinType1,
    "HeatingQC": HeatingQC,
    "KitchenQual": KitchenQual,
    "Functional": Functional,
    "FireplaceQu": FireplaceQu,
    "GarageFinish": GarageFinish,
    "GarageQual": GarageQual,
    "GarageCond": GarageQual
}


def year_grade(year):
    try:
        year = int(year)
        if year >= 2000:
            return 3
        elif year >= 1990:
            return 2
        elif year >= 1980:
            return 1
        else:
            return 0
    except:
        return 0


def complex_process(df):
    new_df = pd.DataFrame()
    df_list = []
    for col in df.columns:
        if col in dummy_list:
            dummy_df = pd.get_dummies(df[col].fillna("NA"), prefix=col)
            df_list.append(dummy_df)
        elif col in numeric_list:
            d = df[col].fillna(0).astype("float32")
            new_df[col] = (d - d.mean()) / d.std()  # 标准化
        elif col in grade_dict:
            d = grade_dict[col]
            new_col = df[col].fillna("NA").map(d)
            new_df[col] = new_col
        else:
            continue
    new_df["YearRemodAdd_Grade"] = df["YearRemodAdd"].apply(year_grade)
    df_list.append(new_df)
    final_df = pd.concat(df_list, axis=1)
    return final_df


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
    train_data = train.drop(['Id', 'SalePrice'], axis=1)
    test_data = test.drop(['Id'], axis=1)
    all_features = pd.concat((train_data.iloc[:, :], test_data.iloc[:, :]))
    
    # feature_df = simple_process(all_features)
    feature_df = complex_process(all_features)

    n_train = train.shape[0]
    train_df = feature_df[:n_train]
    train_label = train['SalePrice']
    test_df  = feature_df[n_train:]
    test_Id = test['Id']
    feature_list = feature_df.columns.tolist()


    log.info(f'Training Shape: {train_df.shape}, {train_label.shape}')
    log.info(f'Testing  Shape: {test_df.shape}, {test_Id.shape}')
    log.info(f"process data in {t.stop():.1f}s")

    return train_df, train_label, test_df, test_Id, feature_list


if __name__ == "__main__":
    train_df, train_label, test_df, test_Id, feature_list = get_feature_df(config.train_input, config.test_input)
