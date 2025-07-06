import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose

# ========== 字体设置 - 解决中文显示问题 ==========
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
except:
    print("字体设置失败，可能需要手动安装中文字体")

# 数据加载与初步探索
df = pd.read_csv('US-pumpkins.csv')
print("原始数据形状:", df.shape)
print("\n缺失值统计:")
print(df.isnull().sum().sort_values(ascending=False))

# 数据清洗
columns_to_drop = ['Type', 'Grade', 'Sub Variety', 'Environment', 'Unit of Sale',
                  'Quality', 'Condition', 'Appearance', 'Storage', 'Crop', 'Repack',
                  'Trans Mode', 'Unnamed: 24', 'Unnamed: 25', 'Origin District']
df_cleaned = df.drop(columns=columns_to_drop)

# 日期格式转换
df_cleaned['Date'] = pd.to_datetime(df_cleaned['Date'], format='%m/%d/%y')

# 统一分类值格式
size_mapping = {'sml': 'small', 'med': 'medium', 'lge': 'large', 'xlge': 'xlarge',
                'jbo': 'jumbo', 'exjbo': 'exjumbo', 'med-lge': 'med-large'}
df_cleaned['Item Size'] = df_cleaned['Item Size'].replace(size_mapping)

# 处理异常价格值
Q1 = df_cleaned['High Price'].quantile(0.25)
Q3 = df_cleaned['High Price'].quantile(0.75)
IQR = Q3 - Q1
price_upper_bound = Q3 + 1.5 * IQR
df_cleaned = df_cleaned[df_cleaned['High Price'] <= price_upper_bound]

# 特征工程
df_cleaned['Avg_Price'] = (df_cleaned['Low Price'] + df_cleaned['High Price']) / 2
df_cleaned['Year'] = df_cleaned['Date'].dt.year
df_cleaned['Month'] = df_cleaned['Date'].dt.month
df_cleaned['Week'] = df_cleaned['Date'].dt.isocalendar().week
df_cleaned['DayOfYear'] = df_cleaned['Date'].dt.dayofyear

# 创建本地标志
city_state_map = {
    'BALTIMORE': 'MARYLAND',
    'ATLANTA': 'GEORGIA',
    'BOSTON': 'MASSACHUSETTS',
    'CHICAGO': 'ILLINOIS',
    'COLUMBIA': 'SOUTH CAROLINA'
}
df_cleaned['Is_Local'] = df_cleaned.apply(
    lambda row: 1 if str(row['Origin']).upper() == city_state_map.get(str(row['City Name']).upper(), '') else 0,
    axis=1
)

# 创建标准化价格（每立方英尺）
def calculate_volume(package):
    if isinstance(package, str):
        if '24 inch bins' in package:
            return np.pi * (12**2) * 24
        elif '36 inch bins' in package:
            return np.pi * (18**2) * 36
        elif '50 lb sacks' in package:
            return 50
        elif '1 1/9 bushel cartons' in package:
            return 1.111 * 1.2445
        elif '1/2 bushel cartons' in package:
            return 0.5 * 1.2445
    return 1

df_cleaned['Volume'] = df_cleaned['Package'].apply(calculate_volume)
df_cleaned['Price_per_cuft'] = df_cleaned['Avg_Price'] / df_cleaned['Volume']

print("\n清洗后数据形状:", df_cleaned.shape)

# ========== 可视化分析 - 不保存图片 ==========

# 1. 品种对价格的影响
plt.figure(figsize=(12, 8))

# 选择所有有足够样本的品种
variety_counts = df_cleaned['Variety'].value_counts()
valid_varieties = variety_counts[variety_counts > 10].index  # 至少有10个样本

# 按价格排序
variety_prices = df_cleaned[df_cleaned['Variety'].isin(valid_varieties)]
variety_prices = variety_prices.groupby('Variety')['Price_per_cuft'].median().sort_values(ascending=False)

# 创建图表
ax = variety_prices.plot(kind='bar', color='darkorange')
plt.title('南瓜品种价格比较（每立方英尺）', fontsize=14)
plt.ylabel('价格 ($/cuft)', fontsize=12)
plt.xlabel('品种', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for i, v in enumerate(variety_prices):
    ax.text(i, v+0.5, f"${v:.2f}", ha='center', fontsize=9)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 2. 大小对价格的影响
plt.figure(figsize=(10, 6))

# 动态获取大小顺序
size_order = df_cleaned['Item Size'].value_counts().index

# 计算价格中位数
size_prices = df_cleaned.groupby('Item Size')['Price_per_cuft'].median().reindex(size_order)

# 过滤有效数据
size_prices = size_prices.dropna()

# 创建图表
ax = size_prices.plot(kind='bar', color='sandybrown')
plt.title('南瓜大小对价格的影响', fontsize=14)
plt.ylabel('价格 ($/cuft)', fontsize=12)
plt.xlabel('大小分类', fontsize=12)
plt.grid(axis='y', linestyle='--', alpha=0.7)

# 添加数值标签
for i, v in enumerate(size_prices):
    ax.text(i, v+0.3, f"${v:.2f}", ha='center', fontsize=9)

plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()

# 3. 时间序列分析（月度）
plt.figure(figsize=(10, 6))
monthly_prices = df_cleaned.groupby('Month')['Price_per_cuft'].mean()
monthly_prices.plot(marker='o', color='peru')
plt.title('南瓜价格月度趋势')
plt.ylabel('平均价格 ($/cuft)')
plt.xlabel('月份')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.tight_layout()
plt.show()

# 4. 季节性分解
plt.figure(figsize=(10, 6))
df_ts = df_cleaned.set_index('Date')['Price_per_cuft'].resample('W').mean().ffill()
decomposition = seasonal_decompose(df_ts, model='additive', period=52)
decomposition.trend.plot(color='brown')
plt.title('南瓜价格长期趋势')
plt.ylabel('价格 ($/cuft)')
plt.grid(True)
plt.tight_layout()
plt.show()

# 5. 本地vs外地价格比较
plt.figure(figsize=(8, 6))
local_prices = df_cleaned.groupby('Is_Local')['Price_per_cuft'].mean()
local_prices.plot(kind='bar', color=['skyblue', 'lightgreen'])
plt.title('本地vs外地南瓜价格比较')
plt.ylabel('平均价格 ($/cuft)')
plt.xticks([0, 1], ['外地', '本地'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 6. 包装类型价格分布
plt.figure(figsize=(10, 6))
package_prices = df_cleaned.groupby('Package')['Price_per_cuft'].median().nlargest(5)
package_prices.plot(kind='barh', color='goldenrod')
plt.title('不同包装类型的价格比较')
plt.xlabel('价格 ($/cuft)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.show()

# 7. 品种-大小交互热图
plt.figure(figsize=(12, 8))
pivot_data = df_cleaned.pivot_table(
    index='Variety',
    columns='Item Size',
    values='Price_per_cuft',
    aggfunc='median'
)
plt.imshow(pivot_data.fillna(0), cmap='YlOrBr', aspect='auto')
plt.colorbar(label='价格 ($/cuft)')
plt.title('品种与大小的价格热图')
plt.xlabel('大小')
plt.ylabel('品种')
plt.xticks(range(len(pivot_data.columns)), pivot_data.columns, rotation=45)
plt.yticks(range(len(pivot_data.index)), pivot_data.index)
plt.tight_layout()
plt.show()

# ========== 预测模型 ==========
# 准备数据
model_df = df_cleaned[['Variety', 'Item Size', 'Package', 'Month', 'Is_Local', 'Price_per_cuft']].copy()
model_df = model_df.dropna()

# 编码分类变量
label_encoders = {}
for col in ['Variety', 'Item Size', 'Package']:
    le = LabelEncoder()
    model_df[col] = le.fit_transform(model_df[col])
    label_encoders[col] = le

# 拆分数据集
X = model_df.drop('Price_per_cuft', axis=1)
y = model_df['Price_per_cuft']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 训练模型
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 评估模型
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
r2 = r2_score(y_test, y_pred)
print(f"\n模型性能评估:")
print(f"RMSE: {rmse:.2f}")
print(f"R²: {r2:.2f}")

# 检查过拟合
y_train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)
print(f"\n训练集表现: RMSE={train_rmse:.2f}, R²={train_r2:.2f}")
print(f"测试集表现: RMSE={rmse:.2f}, R²={r2:.2f}")

# 特征重要性可视化
plt.figure(figsize=(10, 6))
features = X.columns
importances = model.feature_importances_
indices = np.argsort(importances)[::-1]

plt.barh(range(len(indices)), importances[indices], color='sienna', align='center')
plt.yticks(range(len(indices)), [features[i] for i in indices])
plt.xlabel('特征重要性')
plt.title('南瓜价格预测特征重要性')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.show()

# 预测与实际值比较
plt.figure(figsize=(10, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='darkorange')
plt.plot([y.min(), y.max()], [y.min(), y.max()], 'k--', lw=2)
plt.xlabel('实际价格')
plt.ylabel('预测价格')
plt.title('实际价格 vs 预测价格')
plt.grid(True)
plt.tight_layout()
plt.show()
