# 读取 + 清洗 + 特征工程

import pandas as pd
import numpy as np
import warnings

from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings('ignore')


def load_and_clean_data(file_path):
    # 设置临时目录
    import os
    import tempfile
    os.environ['TMP'] = 'C:\\TEMP'
    os.environ['TEMP'] = 'C:\\TEMP'
    tempfile.tempdir = 'C:\\TEMP'

    # 加载数据
    df = pd.read_csv(file_path, encoding='latin1')
    df = df.rename(columns={df.columns[0]: 'City Name'})

    # 移除空列
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    df = df.drop(columns=empty_cols)

    # 移除低方差列
    low_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=low_variance_cols)

    # 选择关键特征
    selected_cols = ['City Name', 'Package', 'Variety', 'Item Size', 'Color',
                     'Origin', 'Date', 'Low Price', 'High Price']
    df = df[selected_cols]

    # 处理日期
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%y')
    df = df[~df['Date'].isnull()]

    # 创建时间特征
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week

    # 季节特征
    df['Season'] = df['Month'].apply(lambda x: 1 if x in [9, 10, 11] else
    2 if x in [12, 1, 2] else
    3 if x in [3, 4, 5] else 4)
    df['Is_Holiday'] = df['Month'].isin([10, 11]).astype(int)

    # 目标变量
    df['Avg_Price'] = (df['Low Price'] + df['High Price']) / 2

    # 处理缺失值
    df['Color'].fillna('UNKNOWN', inplace=True)
    df['Item Size'].fillna('UNKNOWN', inplace=True)
    df['Origin'].fillna('UNKNOWN', inplace=True)

    # 删除异常值
    df = df[(df['Avg_Price'] > 0) & (df['Avg_Price'] < 1000)]
    df = df[df['Year'] >= 2016]

    # 滞后特征
    df = df.sort_values(['Origin', 'Date'])
    df['Prev_Week_Price'] = df.groupby('Origin')['Avg_Price'].shift(1)
    df['Price_Change'] = df.groupby('Origin')['Avg_Price'].pct_change()

    # 区域特征
    region_avg = df.groupby('Origin')['Avg_Price'].mean().reset_index()
    region_avg.columns = ['Origin', 'Region_Avg_Price']
    df = pd.merge(df, region_avg, on='Origin', how='left')

    # 品种稀有度
    variety_count = df['Variety'].value_counts().reset_index()
    variety_count.columns = ['Variety', 'Variety_Count']
    df = pd.merge(df, variety_count, on='Variety', how='left')
    df['Rare_Variety'] = (df['Variety_Count'] < 100).astype(int)

    # 删除缺失值
    df.dropna(subset=['Prev_Week_Price', 'Price_Change'], inplace=True)

    return df


def prepare_features(df):
    # 分类特征编码
    cat_cols = ['City Name', 'Package', 'Variety', 'Item Size', 'Color', 'Origin']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 特征选择
    features = ['Year', 'Month', 'Week', 'Season', 'Is_Holiday',
                'City Name', 'Package', 'Variety', 'Item Size', 'Color',
                'Origin', 'Prev_Week_Price', 'Price_Change',
                'Region_Avg_Price', 'Rare_Variety']

    # 划分数据集
    train = df[df['Year'] == 2016]
    test = df[df['Year'] == 2017]

    X_train = train[features]
    y_train = train['Avg_Price']
    X_test = test[features]
    y_test = test['Avg_Price']

    return X_train, X_test, y_train, y_test