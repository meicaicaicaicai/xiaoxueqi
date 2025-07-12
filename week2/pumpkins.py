import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler

# 1. 加载数据
df = pd.read_csv('US-pumpkins.csv', encoding='latin1')

# 2. 初步数据探索
print("数据维度:", df.shape)
print("\n前5行数据:")
print(df.head())
print("\n数据类型和缺失值:")
print(df.info())
print("\n描述性统计:")
print(df.describe(include='all'))

# 3. 数据清洗
# 选择关键特征
selected_cols = ['City Name', 'Package', 'Variety', 'Item Size', 'Color',
                'Origin', 'Origin District', 'Date', 'Low Price', 'High Price',
                'Mostly Low', 'Mostly High']
df = df[selected_cols]

# 处理日期格式
df['Date'] = pd.to_datetime(df['Date'], format='%m/%d/%y')
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Season'] = df['Month'].apply(lambda x: 'Fall' if x in [9,10,11] else
                                ('Winter' if x in [12,1,2] else
                                ('Spring' if x in [3,4,5] else 'Summer')))

# 创建目标变量 - 平均价格
df['Avg_Price'] = (df['Low Price'] + df['High Price']) / 2

# 处理缺失值
df['Color'].fillna('UNKNOWN', inplace=True)
df['Item Size'].fillna('UNKNOWN', inplace=True)
df['Origin'].fillna('UNKNOWN', inplace=True)

# 删除价格异常记录
df = df[(df['Avg_Price'] > 0) & (df['Avg_Price'] < 1000)]

