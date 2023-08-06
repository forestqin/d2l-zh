import os
import sys
import time
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from IPython.display import display
from pandas.api.types import CategoricalDtype

from category_encoders import MEstimateEncoder
from category_encoders.cat_boost import CatBoostEncoder
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.feature_selection import mutual_info_regression
from xgboost import XGBRegressor
from sklearn.model_selection import KFold, cross_val_score

os.chdir(sys.path[0])

features_nom = ["MSSubClass", "MSZoning", "Street", "Alley", "LandContour", "LotConfig", "Neighborhood", "Condition1", "Condition2", 
                "BldgType", "HouseStyle", "RoofStyle", "RoofMatl", "Exterior1st", "Exterior2nd", "MasVnrType", "Foundation", "Heating", 
                "CentralAir", "GarageType", "MiscFeature", "SaleType", "SaleCondition"]


# The ordinal (ordered) categorical features 

# Pandas calls the categories "levels"
five_levels = ["Po", "Fa", "TA", "Gd", "Ex"]
ten_levels = list(range(10))

ordered_levels = {
    "OverallQual": ten_levels,
    "OverallCond": ten_levels,
    "ExterQual": five_levels,
    "ExterCond": five_levels,
    "BsmtQual": five_levels,
    "BsmtCond": five_levels,
    "HeatingQC": five_levels,
    "KitchenQual": five_levels,
    "FireplaceQu": five_levels,
    "GarageQual": five_levels,
    "GarageCond": five_levels,
    "PoolQC": five_levels,
    "LotShape": ["Reg", "IR1", "IR2", "IR3"],
    "LandSlope": ["Sev", "Mod", "Gtl"],
    "BsmtExposure": ["No", "Mn", "Av", "Gd"],
    "BsmtFinType1": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "BsmtFinType2": ["Unf", "LwQ", "Rec", "BLQ", "ALQ", "GLQ"],
    "Functional": ["Sal", "Sev", "Maj1", "Maj2", "Mod", "Min2", "Min1", "Typ"],
    "GarageFinish": ["Unf", "RFn", "Fin"],
    "PavedDrive": ["N", "P", "Y"],
    "Utilities": ["NoSeWa", "NoSewr", "AllPub"],
    "CentralAir": ["N", "Y"],
    "Electrical": ["Mix", "FuseP", "FuseF", "FuseA", "SBrkr"],
    "Fence": ["MnWw", "GdWo", "MnPrv", "GdPrv"],
}

# Add a None level for missing values
ordered_levels = {key: ["None"] + value for key, value in ordered_levels.items()}

cluster_features = [
    "LotArea",
    "TotalBsmtSF",
    "FirstFlrSF",
    "SecondFlrSF",
    "GrLivArea",
]

pca_features = [
    "GarageArea",
    "YearRemodAdd",
    "TotalBsmtSF",
    "GrLivArea",
]

# def load_data():
#     # Read data
#     data_dir = Path("./data/kaggle_house_price/")
#     df_train = pd.read_csv(data_dir / "kaggle_house_pred_train.csv", index_col="Id")
#     df_test = pd.read_csv(data_dir / "kaggle_house_pred_test.csv", index_col="Id")
#     # Merge the splits so we can process them together
#     df = pd.concat([df_train, df_test])
#     # Preprocessing
#     df = clean(df)
#     df = encode(df)
#     df = impute(df)
#     # Reform splits
#     df_train = df.loc[df_train.index, :]
#     df_test = df.loc[df_test.index, :]
#     return df_train, df_test

def clean(df):
    df["Exterior2nd"] = df["Exterior2nd"].replace({"Brk Cmn": "BrkComm"})
    # Some values of GarageYrBlt are corrupt, so we'll replace them
    # with the year the house was built
    df["GarageYrBlt"] = df["GarageYrBlt"].where(df.GarageYrBlt <= 2010, df.YearBuilt)
    # Names beginning with numbers are awkward to work with
    df.rename(columns={
        "1stFlrSF": "FirstFlrSF",
        "2ndFlrSF": "SecondFlrSF",
        "3SsnPorch": "Threeseasonporch",
    }, inplace=True,
    )
    return df

def encode(df):
    # Nominal categories
    for name in features_nom:
        df[name] = df[name].astype("category")
        # Add a None category for missing values
        if "None" not in df[name].cat.categories:
            df[name].cat.add_categories("None", inplace=True)
    # Ordinal categories
    for name, levels in ordered_levels.items():
        df[name] = df[name].astype(CategoricalDtype(levels,
                                                    ordered=True))
    return df

def impute(df):
    # Create a new column "Imputed" initialized with zeros
    df["Imputed"] = 0
    
    # Perform the imputation
    for name in df.select_dtypes("number"):
        if df[name].isnull().sum() > 0:
            # Impute missing values
            df[name].fillna(value=0, inplace=True)
            # Set "Imputed" column to 1 for imputed rows
            df.loc[df[name].isnull(), "Imputed"] = 1
    for name in df.select_dtypes(["object"]):
        if df[name].isnull().sum() > 0:
            # Impute missing values
            df[name].fillna(value="None", inplace=True)
            # Set "Imputed" column to 1 for imputed rows
            df.loc[df[name].isnull(), "Imputed"] = 1
    return df

# Step 2 - Feature Utility Scores

def make_mi_scores(X, y):
    X = X.copy()
    for colname in X.select_dtypes(["object", "category"]):
        X[colname], _ = X[colname].factorize()
    # All discrete features should now have integer dtypes
    discrete_features = [pd.api.types.is_integer_dtype(t) for t in X.dtypes]
    mi_scores = mutual_info_regression(X, y, discrete_features=discrete_features, random_state=0)
    mi_scores = pd.Series(mi_scores, name="MI Scores", index=X.columns)
    mi_scores = mi_scores.sort_values(ascending=False)
    return mi_scores

def plot_mi_scores(scores):
    scores = scores.sort_values(ascending=True)
    width = np.arange(len(scores))
    ticks = list(scores.index)
    plt.barh(width, scores)
    plt.yticks(width, ticks)
    plt.title("Mutual Information Scores")

def drop_uninformative(df, mi_scores):
    return df.loc[:, mi_scores > 0.0000000001]

# Step 3 - Create Features 

def label_encode(df):
    X = df.copy()
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    return X

def mathematical_transforms(df):
    X = pd.DataFrame()  # dataframe to hold new features
    X["LivLotRatio"] = df.GrLivArea / df.LotArea
    X["Spaciousness"] = (df.FirstFlrSF + df.SecondFlrSF) / df.TotRmsAbvGrd
    # This feature ended up not helping performance
    # X["TotalOutsideSF"] = \
    #     df.WoodDeckSF + df.OpenPorchSF + df.EnclosedPorch + \
    #     df.Threeseasonporch + df.ScreenPorch
    return X


def interactions(df):
    X = pd.get_dummies(df.BldgType, prefix="Bldg")
    X = X.mul(df.GrLivArea, axis=0)
    return X


def counts(df):
    X = pd.DataFrame()
    X["PorchTypes"] = df[[
        "WoodDeckSF",
        "OpenPorchSF",
        "EnclosedPorch",
        "Threeseasonporch",
        "ScreenPorch",
    ]].gt(0.0).sum(axis=1)
    return X


def break_down(df):
    X = pd.DataFrame()
    X["MSClass"] = df.MSSubClass.str.split("_", n=1, expand=True)[0]
    return X

def group_transforms(df):
    X = pd.DataFrame()
    X["MedNhbdArea"] = df.groupby("Neighborhood")["GrLivArea"].transform("median")
    return X
# Transformations to try
def OverallQuall(df):
    X = pd.DataFrame()
    X["QualCond"] = df.OverallQual.factorize()[0] * df.OverallCond.factorize()[0]
    return X

# Converting square features to linear
def to_root_square(df):
    X = pd.DataFrame()
    X["LotArea_root"] = np.sqrt(df.LotArea)
    X["BsmtFinSF1_root"] = np.sqrt(df.BsmtFinSF1)
    X["BsmtFinSF2_root"] = np.sqrt(df.BsmtFinSF2)
    X["BsmtUnfSF_root"] = np.sqrt(df.BsmtUnfSF)
    X["TotalBsmtSF_root"] = np.sqrt(df.TotalBsmtSF)
    X["FirstFlrSF_root"] = np.sqrt(df.FirstFlrSF)
    X["SecondFlrSF_root"] = np.sqrt(df.SecondFlrSF)
    X["LowQualFinSF_root"] = np.sqrt(df.LowQualFinSF)
    X["GrLivArea_root"] = np.sqrt(df.GrLivArea)
    X["GarageArea_root"] = np.sqrt(df.GarageArea)
    X["WoodDeckSF_root"] = np.sqrt(df.WoodDeckSF)
    X["OpenPorchSF_root"] = np.sqrt(df.OpenPorchSF)
    X["EnclosedPorch_root"] = np.sqrt(df.EnclosedPorch)
    X["Threeseasonporch_root"] = np.sqrt(df.Threeseasonporch)
    X["ScreenPorch_root"] = np.sqrt(df.ScreenPorch)
    X["MedNhbdArea_root"] = np.sqrt(df.MedNhbdArea)
    X["Spaciousness_root"] = np.sqrt(df.Spaciousness)
    return X


def cluster_labels(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=n_clusters, n_init=50, random_state=0)
    X_new = pd.DataFrame()
    X_new["Cluster"] = kmeans.fit_predict(X_scaled)
    return X_new


def cluster_distance(df, features, n_clusters=20):
    X = df.copy()
    X_scaled = X.loc[:, features]
    X_scaled = (X_scaled - X_scaled.mean(axis=0)) / X_scaled.std(axis=0)
    kmeans = KMeans(n_clusters=20, n_init=50, random_state=0)
    X_cd = kmeans.fit_transform(X_scaled)
    # Label features and join to dataset
    X_cd = pd.DataFrame(
        X_cd, columns=[f"Centroid_{i}" for i in range(X_cd.shape[1])]
    )
    return X_cd

def apply_pca(X, standardize=True):
    # Standardize
    if standardize:
        X = (X - X.mean(axis=0)) / X.std(axis=0)
    # Create principal components
    pca = PCA()
    X_pca = pca.fit_transform(X)
    # Convert to dataframe
    component_names = [f"PC{i+1}" for i in range(X_pca.shape[1])]
    X_pca = pd.DataFrame(X_pca, columns=component_names)
    # Create loadings
    loadings = pd.DataFrame(
        pca.components_.T,  # transpose the matrix of loadings
        columns=component_names,  # so the columns are the principal components
        index=X.columns,  # and the rows are the original features
    )
    return pca, X_pca, loadings


def plot_variance(pca, width=8, dpi=100):
    # Create figure
    fig, axs = plt.subplots(1, 2)
    n = pca.n_components_
    grid = np.arange(1, n + 1)
    # Explained variance
    evr = pca.explained_variance_ratio_
    axs[0].bar(grid, evr)
    axs[0].set(
        xlabel="Component", title="% Explained Variance", ylim=(0.0, 1.0)
    )
    # Cumulative Variance
    cv = np.cumsum(evr)
    axs[1].plot(np.r_[0, grid], np.r_[0, cv], "o-")
    axs[1].set(
        xlabel="Component", title="% Cumulative Variance", ylim=(0.0, 1.0)
    )
    # Set up figure
    fig.set(figwidth=8, dpi=100)
    return axs

def pca_inspired(df):
    X = pd.DataFrame()
    X["Feature1"] = df.GrLivArea + df.TotalBsmtSF
    X["Feature2"] = df.YearRemodAdd * df.TotalBsmtSF
    return X

def pca_components(df, features):
    X = df.loc[:, features]
    _, X_pca, _ = apply_pca(X)
    return X_pca



def indicate_outliers(df):
    X_new = pd.DataFrame()
    X_new["Outlier"] = (df.Neighborhood == "Edwards") & (df.SaleCondition == "Partial")
    return X_new

class CrossFoldEncoder:
    def __init__(self, encoder, **kwargs):
        self.encoder_ = encoder
        self.kwargs_ = kwargs  # keyword arguments for the encoder
        self.cv_ = KFold(n_splits=5)

    # Fit an encoder on one split and transform the feature on the
    # other. Iterating over the splits in all folds gives a complete
    # transformation. We also now have one trained encoder on each
    # fold.
    def fit_transform(self, X, y, cols):
        self.fitted_encoders_ = []
        self.cols_ = cols
        X_encoded = []
        for idx_encode, idx_train in self.cv_.split(X):
            fitted_encoder = self.encoder_(cols=cols, **self.kwargs_)
            fitted_encoder.fit(
                X.iloc[idx_encode, :], y.iloc[idx_encode],
            )
            X_encoded.append(fitted_encoder.transform(X.iloc[idx_train, :])[cols])
            self.fitted_encoders_.append(fitted_encoder)
        X_encoded = pd.concat(X_encoded)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded

    # To transform the test data, average the encodings learned from
    # each fold.
    def transform(self, X):
        from functools import reduce

        X_encoded_list = []
        for fitted_encoder in self.fitted_encoders_:
            X_encoded = fitted_encoder.transform(X)
            X_encoded_list.append(X_encoded[self.cols_])
        X_encoded = reduce(
            lambda x, y: x.add(y, fill_value=0), X_encoded_list
        ) / len(X_encoded_list)
        X_encoded.columns = [name + "_encoded" for name in X_encoded.columns]
        return X_encoded
    

def create_features(df, y, df_test=None):
    X = df.copy()
    mi_scores = make_mi_scores(X, y)

    if df_test is not None:
        X_test = df_test.copy()
        X = pd.concat([X, X_test])

    # Lesson 2 - Mutual Information
    X = drop_uninformative(X, mi_scores)

    # Lesson 3 - Transformations
    X = X.join(mathematical_transforms(X))
    X = X.join(interactions(X))
    X = X.join(counts(X))
    # X = X.join(break_down(X))
    X = X.join(group_transforms(X))
    
    """
    To try new transformations after this:
    """
    X = X.join(OverallQuall(X))
#     X = X.join(to_root_square(X))
    
    
    # Lesson 4 - Clustering
    # X = X.join(cluster_labels(X, cluster_features, n_clusters=20))
    # X = X.join(cluster_distance(X, cluster_features, n_clusters=20))

    # Lesson 5 - PCA
    X = X.join(pca_inspired(X))
    # X = X.join(pca_components(X, pca_features))
    # X = X.join(indicate_outliers(X))

    X = label_encode(X)

    # Reform splits
    if df_test is not None:
        X_test = X.loc[df_test.index, :]
        X.drop(df_test.index, inplace=True)

    # Lesson 6 - Target Encoder
    encoder = CrossFoldEncoder(MEstimateEncoder, m=1)
    X = X.join(encoder.fit_transform(X, y, cols=["MSSubClass"]))
    if df_test is not None:
        X_test = X_test.join(encoder.transform(X_test))

    if df_test is not None:
        return X, X_test
    else:
        return X


def get_feature_df(create_feature=True):
    train_input = "./data/kaggle_house_price/kaggle_house_pred_train.csv"
    test_input = "./data/kaggle_house_price/kaggle_house_pred_test.csv"

    t = time.time()
    train = pd.read_csv(train_input, index_col="Id")
    train_df = train.drop(['SalePrice'], axis=1)
    train_y = np.log1p(train['SalePrice'])
    # feature_list = train_df.columns.tolist()

    test_df = pd.read_csv(test_input, index_col="Id")
    test_Id = test_df.index
    # Preprocessing
    df = pd.concat([train_df, test_df])
    df = clean(df)
    df = encode(df)
    df = impute(df)
    # Reform splits
    train_X = df.loc[train_df.index, :]
    test_X = df.loc[test_df.index, :]
    print(f'Training Shape: {train_X.shape}, {train_y.shape}')
    print(f'Testing  Shape: {test_X.shape}, {test_Id.shape}')
    print(f"process data in {time.time()-t:.1f}s")

    if create_feature:
        print("create feature...")
        train_X, test_X = create_features(train_X, train_y, test_X)
        print(f'Training Shape(create feature): {train_X.shape}, {train_y.shape}')
        print(f'Testing  Shape(create feature): {test_X.shape}, {test_Id.shape}')
        print(f"create feature in {time.time()-t:.1f}s")

    train = train_X.join(train_y)
    train.to_csv(train_input.replace(".csv", "_features.csv"))
    test_X.to_csv(test_input.replace(".csv", "_features.csv"))

    return train_X, train_y, test_X, test_Id

def get_feature_df_cached():
    train_input = "./data/kaggle_house_price/kaggle_house_pred_train_features.csv"
    test_input = "./data/kaggle_house_price/kaggle_house_pred_test_features.csv"

    train_X = pd.read_csv(train_input, index_col="Id")
    train_y = train_X['SalePrice']
    train_X.drop('SalePrice', axis=1, inplace=True)

    test_X = pd.read_csv(test_input)
    test_Id = test_X["Id"]
    test_X.drop("Id", axis=1, inplace=True)

    print(f'Training Shape(create feature): {train_X.shape}, {train_y.shape}')
    print(f'Testing  Shape(create feature): {test_X.shape}, {test_Id.shape}')

    return train_X, train_y, test_X, test_Id

def score_dataset(X, y, model):
    for colname in X.select_dtypes(["category"]):
        X[colname] = X[colname].cat.codes
    # Metric for Housing competition is RMSLE (Root Mean Squared Log Error)
    log_y = np.log(y)
    score = cross_val_score(
        model, X, log_y, cv=5, scoring="neg_mean_squared_error",
    )
    score = -1 * score.mean()
    score = np.sqrt(score)
    return score

def base_algorithm():
    train_df, train_label, test_df, test_Id = get_feature_df(False)
    X = train_df.copy()
    y = train_label

    xgb = XGBRegressor()
    baseline_score = score_dataset(X, y, xgb)
    print(f"Baseline score: {baseline_score:.5f} RMSLE")

    xgb = XGBRegressor()
    mi_scores = make_mi_scores(X, y)
    X = drop_uninformative(X, mi_scores)
    score = score_dataset(X, y, xgb)
    print(f"Make MI score: {score:.5f} RMSLE")


if __name__ == "__main__":
    # base_algorithm()
    
    train_X, train_y, test_X, test_Id = get_feature_df()
    xgb = XGBRegressor()
    score = score_dataset(train_X, train_y, xgb)
    print(f"Make Feature score: {score:.5f} RMSLE")
    

    # # Step 5 - Train Model and Create Submissions
    # X_train, X_test = create_features(df_train, df_test)
    # y_train = df_train.loc[:, "SalePrice"]

    # xgb = XGBRegressor(**xgb_params)
    # # XGB minimizes MSE, but competition loss is RMSLE
    # # So, we need to log-transform y to train and exp-transform the predictions
    # xgb.fit(X_train, np.log(y))
    # predictions = np.exp(xgb.predict(X_test))

    # output = pd.DataFrame({'Id': X_test.index, 'SalePrice': predictions})
    # output.to_csv('output/submission_xgb_optuna.csv', index=False)
    # print("Your submission was successfully saved!")