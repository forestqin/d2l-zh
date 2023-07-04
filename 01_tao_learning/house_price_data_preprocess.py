import os
import sys
import pandas as pd
import numpy as np

os.chdir(sys.path[0])

ignore_list = [
    "Id", "LotFrontage", "Street", "Alley", "Utilities", "Condition2",
    "YearBuilt", "RoofMatl", "BsmtFinType2", "BsmtFinSF2", "BsmtUnfSF",
    "Electrical", "PoolArea", "PoolQC", "MiscVal", "MoSold", "YrSold"
]

dummy_list = [
    "MSSubClass", "MSZoning", "LotConfig", "Neighborhood", "Condition1",
    "BldgType", "HouseStyle", "RoofStyle", "Exterior1st", "Exterior2nd",
    "MasVnrType", "Foundation", "Heating", "CentralAir", "GarageType",
    "PavedDrive", "Fence", "MiscFeature", "SaleType", "SaleCondition"
]

numeric_list = [
    "LotArea", "OverallQual", "OverallCond", "YearRemodAdd", "MasVnrArea",
    "BsmtFinSF1", "TotalBsmtSF", "1stFlrSF", "2ndFlrSF", "LowQualFinSF",
    "GrLivArea", "BsmtFullBath", "BsmtHalfBath", "FullBath", "HalfBath",
    "BedroomAbvGr", "KitchenAbvGr", "TotRmsAbvGrd", "Fireplaces",
    "GarageYrBlt", "GarageCars", "GarageArea", "WoodDeckSF", "OpenPorchSF",
    "EnclosedPorch", "3SsnPorch", "ScreenPorch"
]

LotShape = {"Reg": 4, "IR1": 3, "IR2": 2, "IR3": 1}
LandContour = {"Low": 1, "Bnk": 2, "HLS": 3, "Lvl": 4}
LandSlope = {"Sev": 1, "Mod": 3, "Gtl": 5}
ExterQual = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
BsmtQual = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
BsmtExposure = {"NA": 0, "No": 1, "Mn": 2, "Av": 3, "Gd": 4}
BsmtFinType1 = {
    "NA": 0,
    "Unf": 1,
    "LwQ": 2,
    "Rec": 3,
    "BLQ": 4,
    "ALQ": 5,
    "GLQ": 6
}
HeatingQC = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
KitchenQual = {"Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
Functional = {
    "Sal": 1,
    "Sev": 2,
    "Maj2": 3,
    "Maj1": 4,
    "Mod": 5,
    "Min2": 6,
    "Min1": 7,
    "Typ": 8
}
FireplaceQu = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
GarageFinish = {"NA": 0, "Unf": 1, "RFn": 2, "Fin": 3}
GarageQual = {"NA": 0, "Po": 1, "Fa": 2, "TA": 3, "Gd": 4, "Ex": 5}
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

target = "SalePrice"


def preprocess(df, is_train):
    # df['UnitPrice'] = df['SalePrice'] / df['LotArea']
    new_df = pd.DataFrame()
    # target_col = None
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
    if is_train:  # train dataset: target column
        new_df[target] = df[target].fillna("NA")
    else:  # test dataset: Id column
        new_df["Id"] = df["Id"]
    df_list.append(new_df)
    final_df = pd.concat(df_list, axis=1)
    return final_df


if __name__ == "__main__":
    BASE_PATH = "./data/kaggle_house_price"

    # train
    input_path = os.path.join(BASE_PATH, "kaggle_house_pred_train.csv")
    df_train = pd.read_csv(input_path)
    df_train2 = preprocess(df_train, True)
    print(df_train2.head())
    output_path = os.path.join(BASE_PATH,
                               "kaggle_house_pred_train_processed.csv")
    df_train2.to_csv(output_path, sep="\t", index=False)

    print("finished train preprocessing...")

    # test
    input_path = os.path.join(BASE_PATH, "kaggle_house_pred_test.csv")
    df_test = pd.read_csv(input_path)
    df_test2 = preprocess(df_test, False)
    print(df_test2.head())
    output_path = os.path.join(BASE_PATH,
                               "kaggle_house_pred_test_processed.csv")
    df_test2.to_csv(output_path, sep="\t", index=False)
    # for col in df_test2.columns:
    #     print(col)
    print("finished test preprocessing...")