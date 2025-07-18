# 读取 + 清洗 + 特征工程
# 1.导入
# 2.load_and_clean_data()
#     读取原始数据
#     清洗（空列、低方差、异常值）
#     特征工程（时间、滞后、区域、稀有度）
#     返回干净 DataFrame
# 3.prepare_features()
#     类别编码
#     特征选择
#     按年切训练/测试
# 4.__main__
#     独立演示
#     保存中间结果

# pandas做表格数据的核心库
import pandas as pd
import numpy as np
import warnings
# LabelEncoder 把字符串类别变成整数，树模型可直接吃
from sklearn.preprocessing import LabelEncoder
# filterwarnings('ignore')：把“黄色提示”全部静音
warnings.filterwarnings('ignore')


def load_and_clean_data(file_path):
    # 设置临时目录
    import os
    import tempfile
    # os.environ：修改进程级环境变量，告诉第三方库把临时文件放到 C:\TEMP
    os.environ['TMP'] = 'C:\\TEMP'
    os.environ['TEMP'] = 'C:\\TEMP'
    # tempfile.tempdir：pandas.read_csv 读大文件时可能解压到临时目录
    tempfile.tempdir = 'C:\\TEMP'

    # 加载数据
    # latin1 解决非 UTF-8 字符乱码
    df = pd.read_csv(file_path, encoding='latin1')
    # 把第一列强行命名为 City Name
    df = df.rename(columns={df.columns[0]: 'City Name'})

    # 移除空列
    # 全空直接扔
    empty_cols = [col for col in df.columns if df[col].isnull().all()]
    df = df.drop(columns=empty_cols)

    # 移除低方差列
    # nunique() <= 1 即列内只有 1 个唯一值 → 方差 = 0 → 树模型/线性模型学不到任何东西
    low_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
    df = df.drop(columns=low_variance_cols)

    # 选择关键特征
    # 只拿后续需要的 9 个字段
    selected_cols = ['City Name', 'Package', 'Variety', 'Item Size', 'Color',
                     'Origin', 'Date', 'Low Price', 'High Price']
    df = df[selected_cols]

    # 处理日期
    df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%y')
    # isnull() 筛掉解析失败的行
    df = df[~df['Date'].isnull()]

    # 创建时间特征
    df['Year'] = df['Date'].dt.year
    df['Month'] = df['Date'].dt.month
    df['Week'] = df['Date'].dt.isocalendar().week

    # 季节特征
    df['Season'] = df['Month'].apply(lambda x: 1 if x in [9, 10, 11] else
    2 if x in [12, 1, 2] else
    3 if x in [3, 4, 5] else 4)
    # 10、11月万圣节，需求高峰
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

    # 滞后特征：一个地区南瓜本周价格往往跟上周价格强相关，若上周暴涨 20%，本周可能回调
    df = df.sort_values(['Origin', 'Date'])
    # 同产地上一周均价，用来捕捉价格惯性
    df['Prev_Week_Price'] = df.groupby('Origin')['Avg_Price'].shift(1)
    # 同产地周环比，用来动量特征
    df['Price_Change'] = df.groupby('Origin')['Avg_Price'].pct_change()

    # 区域特征
    region_avg = df.groupby('Origin')['Avg_Price'].mean().reset_index()
    # Region_Avg_Price：产地整体价格水平
    region_avg.columns = ['Origin', 'Region_Avg_Price']
    df = pd.merge(df, region_avg, on='Origin', how='left')

    # 品种稀有度
    variety_count = df['Variety'].value_counts().reset_index()
    variety_count.columns = ['Variety', 'Variety_Count']
    df = pd.merge(df, variety_count, on='Variety', how='left')
    # Rare_Variety：稀有品种标记，可能价格更高
    df['Rare_Variety'] = (df['Variety_Count'] < 100).astype(int)

    # 删除缺失值
    df.dropna(subset=['Prev_Week_Price', 'Price_Change'], inplace=True)

    return df


def prepare_features(df):
    # 分类特征编码
    # 把字符串变成整数编号
    cat_cols = ['City Name', 'Package', 'Variety', 'Item Size', 'Color', 'Origin']
    for col in cat_cols:
        le = LabelEncoder()
        df[col] = le.fit_transform(df[col].astype(str))

    # 特征选择，保留16个数值/类别特征
    features = ['Year', 'Month', 'Week', 'Season', 'Is_Holiday',
                'City Name', 'Package', 'Variety', 'Item Size', 'Color',
                'Origin', 'Prev_Week_Price', 'Price_Change',
                'Region_Avg_Price', 'Rare_Variety']

    # 划分数据集，16学规律，17做预测
    train = df[df['Year'] == 2016]
    test = df[df['Year'] == 2017]

    X_train = train[features]
    y_train = train['Avg_Price']
    X_test = test[features]
    y_test = test['Avg_Price']

    return X_train, X_test, y_train, y_test



if __name__ == '__main__':
    import os
    # 绝对路径或相对路径均可
    file_path = os.path.abspath(r'D:\Python\pycharm\xiaoxueqi\week4\data\US-pumpkins.csv')
    df = load_and_clean_data(file_path)
    print("清洗后形状:", df.shape)
    print(df.head())
    print("缺失值统计:\n", df.isnull().sum())
    # 保存中间结果便于验证
    os.makedirs('../output', exist_ok=True)
    df.to_csv('../output/cleaned.csv', index=False)