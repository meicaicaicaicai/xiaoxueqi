# 编码、缩放、数据集划分

import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from configuration import RANDOM_STATE
from data_analysis import clean, engineer, load_raw

FEATURES = ['Year', 'Month', 'Week', 'Season', 'Is_Holiday',
            'City Name', 'Package', 'Variety', 'Item Size', 'Color',
            'Origin', 'Prev_Week_Price', 'Price_Change',
            'Region_Avg_Price', 'Rare_Variety']

def build_dataset():
    df = engineer(clean(load_raw()))
    df = df.dropna(subset=['Prev_Week_Price', 'Price_Change'])

    # 编码
    cat_cols = ['City Name', 'Package', 'Variety', 'Item Size', 'Color', 'Origin']
    for col in cat_cols:
        df[col] = LabelEncoder().fit_transform(df[col].astype(str))

    train_df = df[df['Year'] == 2016]
    test_df  = df[df['Year'] == 2017]

    X_train = train_df[FEATURES].values
    y_train = train_df['Avg_Price'].values
    X_test  = test_df[FEATURES].values
    y_test  = test_df['Avg_Price'].values

    # 缩放
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test  = scaler.transform(X_test)

    return X_train, y_train, X_test, y_test, FEATURES, scaler