# 读取 + 清洗 + 特征工程

import pandas as pd
import numpy as np
from configuration import DATA_PATH

def load_raw() -> pd.DataFrame:
    return pd.read_csv(DATA_PATH, encoding='latin1')

def clean(df: pd.DataFrame) -> pd.DataFrame:
    # 修复首列列名
    df = df.rename(columns={df.columns[0]: 'City Name'})

    # 删除全空 / 低方差列
    df = df.drop(columns=[c for c in df.columns if df[c].isnull().all() or df[c].nunique() <= 1])

    # 选择核心列
    use = ['City Name', 'Package', 'Variety', 'Item Size', 'Color',
           'Origin', 'Date', 'Low Price', 'High Price']
    df = df[use]

    # 日期处理
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%y')
    df = df[~df['Date'].isnull()]
    df['Year']  = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week']  = df['Date'].dt.isocalendar().week

    # 目标变量
    df['Avg_Price'] = (df['Low Price'] + df['High Price']) / 2

    # 缺失值
    df['Color']     = df['Color'].fillna('UNKNOWN')
    df['Item Size'] = df['Item Size'].fillna('UNKNOWN')
    df['Origin']    = df['Origin'].fillna('UNKNOWN')

    # 异常值
    df = df[(df['Avg_Price'] > 0) & (df['Avg_Price'] < 1000)]

    # 年份过滤
    df = df[df['Year'] >= 2016]

    return df.reset_index(drop=True)

def engineer(df: pd.DataFrame) -> pd.DataFrame:
    """高级特征工程"""
    # 季节、假期
    df['Season'] = df['Month'].map(lambda m: 1 if m in [9,10,11] else
                                             2 if m in [12,1,2] else
                                             3 if m in [3,4,5] else 4)
    df['Is_Holiday'] = df['Month'].isin([10,11]).astype(int)

    # 滞后特征
    df = df.sort_values(['Origin', 'Date'])
    df['Prev_Week_Price'] = df.groupby('Origin')['Avg_Price'].shift(1)
    df['Price_Change']    = df.groupby('Origin')['Avg_Price'].pct_change()

    # 区域均价
    region_avg = df.groupby('Origin')['Avg_Price'].mean().reset_index(name='Region_Avg_Price')
    df = df.merge(region_avg, on='Origin', how='left')

    # 品种稀有度
    vc = df['Variety'].value_counts().reset_index(name='Variety_Count')
    df = df.merge(vc, left_on='Variety', right_index=True, how='left')
    df['Rare_Variety'] = (df['Variety_Count'] < 100).astype(int)

    return df