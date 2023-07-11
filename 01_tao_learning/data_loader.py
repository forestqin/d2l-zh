def load_data():
    df_train = pd.read_csv(
        "./data/kaggle_house_price/kaggle_house_pred_train_processed.csv",
        sep="\t")
    feature_list = df_train.iloc[:, 1:].columns
    train_features = df_train.iloc[:, :-1].values
    train_labels = df_train.iloc[:, -1].values
    train_X, test_X, train_y, test_y = train_test_split(train_features,
                                                        train_labels,
                                                        test_size=0.25,
                                                        random_state=42)
    print('Training Features Shape:', train_X.shape)
    print('Training Labels Shape:', train_y.shape)
    print('Testing Features Shape:', test_X.shape)
    print('Testing Labels Shape:', test_y.shape)
    return train_X, test_X, train_y, test_y, feature_list