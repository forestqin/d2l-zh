import os
import sys
import pandas as pd
import numpy as np
import config

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


def gen_feature_dataframe(input_path, is_train):
    df = pd.read_csv(input_path)
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

    if is_train:  # train dataset: target column
        new_df[config.target] = df[config.target].fillna("NA")
    else:  # test dataset: Id column
        new_df["Id"] = df["Id"]
    df_list.append(new_df)
    final_df = pd.concat(df_list, axis=1)

    output_path = input_path.replace(".csv", "_etl.csv")
    final_df.to_csv(output_path, index=False)
    print(f"output path: {output_path}")
    return final_df


if __name__ == "__main__":
    train_df = preprocess(config.train_raw_input, is_train=True)
    test_df = preprocess(config.test_raw_input, is_train=False)
