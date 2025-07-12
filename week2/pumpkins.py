import os
import tempfile

# 设置临时目录为纯ASCII路径
os.environ['TMP'] = 'C:\\TEMP'  # 使用简单的纯ASCII路径
os.environ['TEMP'] = 'C:\\TEMP'
tempfile.tempdir = 'C:\\TEMP'

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings

warnings.filterwarnings('ignore')

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_theme(font="SimHei")  # 设置seaborn全局字体

# 1. 加载数据
df = pd.read_csv('US-pumpkins.csv', encoding='latin1')

# 修复第一列名称
df = df.rename(columns={df.columns[0]: 'City Name'})

# 2. 数据清洗 - 移除全为空或几乎为空的列
# 检查并删除全为空的列
empty_cols = [col for col in df.columns if df[col].isnull().all()]
df = df.drop(columns=empty_cols)

# 检查并删除唯一值过少的列
low_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
df = df.drop(columns=low_variance_cols)

print("\n清洗后的列名:", df.columns.tolist())

# 3. 数据清洗与特征工程增强
# 选择关键特征
selected_cols = ['City Name', 'Package', 'Variety', 'Item Size', 'Color',
                 'Origin', 'Date', 'Low Price', 'High Price']
df = df[selected_cols]

# 处理日期格式
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%y')

# 移除无效日期
df = df[~df['Date'].isnull()]
df['Year'] = df['Date'].dt.year
df['Month'] = df['Date'].dt.month
df['Week'] = df['Date'].dt.isocalendar().week

# 高级时间特征
df['Season'] = df['Month'].apply(lambda x: 1 if x in [9, 10, 11] else  # 秋季
2 if x in [12, 1, 2] else  # 冬季
3 if x in [3, 4, 5] else 4)  # 夏季
df['Is_Holiday'] = df['Month'].isin([10, 11]).astype(int)  # 万圣节/感恩季

# 创建目标变量 - 平均价格
df['Avg_Price'] = (df['Low Price'] + df['High Price']) / 2

# 处理缺失值
df['Color'].fillna('UNKNOWN', inplace=True)
df['Item Size'].fillna('UNKNOWN', inplace=True)
df['Origin'].fillna('UNKNOWN', inplace=True)

# 删除价格异常记录
df = df[(df['Avg_Price'] > 0) & (df['Avg_Price'] < 1000)]

# 修复数据不平衡问题 - 删除2014年数据
print("\n年份分布 (清洗前):")
print(df['Year'].value_counts().sort_index())
df = df[df['Year'] >= 2016]  # 移除2014年数据
print("\n年份分布 (清洗后):")
print(df['Year'].value_counts().sort_index())

# 4. 高级特征工程
# 滞后特征创建 (按地区分组)
df = df.sort_values(['Origin', 'Date'])
df['Prev_Week_Price'] = df.groupby('Origin')['Avg_Price'].shift(1)
df['Price_Change'] = df.groupby('Origin')['Avg_Price'].pct_change()

# 区域价格水平特征
region_avg = df.groupby('Origin')['Avg_Price'].mean().reset_index()
region_avg.columns = ['Origin', 'Region_Avg_Price']
df = pd.merge(df, region_avg, on='Origin', how='left')

# 品种稀有度特征
variety_count = df['Variety'].value_counts().reset_index()
variety_count.columns = ['Variety', 'Variety_Count']
df = pd.merge(df, variety_count, on='Variety', how='left')
df['Rare_Variety'] = (df['Variety_Count'] < 100).astype(int)

# 5. 特征编码与转换
# 处理分类特征
cat_cols = ['City Name', 'Package', 'Variety', 'Item Size', 'Color', 'Origin']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))

# 特征选择
features = ['Year', 'Month', 'Week', 'Season', 'Is_Holiday',
            'City Name', 'Package', 'Variety', 'Item Size', 'Color',
            'Origin', 'Prev_Week_Price',
            'Price_Change', 'Region_Avg_Price', 'Rare_Variety']

# 删除滞后特征导致的缺失值
df.dropna(subset=['Prev_Week_Price', 'Price_Change'], inplace=True)

# 6. 时间序列数据划分
print("\n最终年份分布:")
print(df['Year'].value_counts().sort_index())

# 按时间顺序划分
train = df[df['Year'] == 2016]
test = df[df['Year'] == 2017]

X_train = train[features]
y_train = train['Avg_Price']
X_test = test[features]
y_test = test['Avg_Price']

print(f"训练集大小: {X_train.shape}, 测试集大小: {X_test.shape}")

# 7. 特征缩放
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# 8. 模型配置与超参数优化
models = {}

# Gradient Boosting 参数网格
gb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'min_samples_split': [2, 5]
}

# XGBoost 参数网格
xgb_param_grid = {
    'n_estimators': [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth': [3, 5],
    'subsample': [0.8, 1.0]
}

# 时间序列交叉验证
tscv = TimeSeriesSplit(n_splits=5)

# 模型初始化
models['Gradient Boosting'] = GridSearchCV(
    GradientBoostingRegressor(random_state=42),
    param_grid=gb_param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

models['XGBoost'] = GridSearchCV(
    XGBRegressor(objective='reg:squarederror', random_state=42, n_jobs=-1),
    param_grid=xgb_param_grid,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1,
    verbose=1
)

# 9. 模型训练与评估
results = {}
print("\n开始模型训练与优化...")

for name, model in models.items():
    print(f"\n训练 {name} 模型...")

    try:
        # 训练模型
        model.fit(X_train_scaled, y_train)

        # 获取最佳参数
        print(f"{name} 最佳参数: {model.best_params_}")

        # 训练集预测
        train_pred = model.predict(X_train_scaled)

        # 测试集预测
        test_pred = model.predict(X_test_scaled)

        # 计算指标
        train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
        test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
        test_mae = mean_absolute_error(y_test, test_pred)
        test_r2 = r2_score(y_test, test_pred)

        # 存储结果
        results[name] = {
            'model': model.best_estimator_,
            'train_rmse': train_rmse,
            'test_rmse': test_rmse,
            'test_mae': test_mae,
            'test_r2': test_r2,
            'test_pred': test_pred
        }

        print(f"{name} 性能:")
        print(f"  训练RMSE: {train_rmse:.2f}")
        print(f"  测试RMSE: {test_rmse:.2f}")
        print(f"  测试MAE: {test_mae:.2f}")
        print(f"  测试R²: {test_r2:.4f}")

    except Exception as e:
        print(f"训练 {name} 时出错: {str(e)}")
        results[name] = {'error': str(e)}

# 10. 模型比较与可视化
if results:
    metrics_data = []
    for name, data in results.items():
        if 'test_r2' in data:
            metrics_data.append({
                'Model': name,
                'MAE': data['test_mae'],
                'RMSE': data['test_rmse'],
                'R2': data['test_r2']
            })

    metrics_df = pd.DataFrame(metrics_data)

    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    sns.barplot(x='R2', y='Model', data=metrics_df.sort_values('R2', ascending=False))
    plt.title('模型R²比较')
    plt.xlim(0, 1)

    plt.subplot(1, 3, 2)
    sns.barplot(x='RMSE', y='Model', data=metrics_df.sort_values('RMSE'))
    plt.title('模型RMSE比较')

    plt.subplot(1, 3, 3)
    sns.barplot(x='MAE', y='Model', data=metrics_df.sort_values('MAE'))
    plt.title('模型MAE比较')

    plt.tight_layout()
    plt.show()

    # 11. 最佳模型分析
    best_model_name = max(results, key=lambda x: results[x]['test_r2'])
    best_model = results[best_model_name]['model']
    print(f"\n最佳模型: {best_model_name} (R²={results[best_model_name]['test_r2']:.4f})")

    # 特征重要性可视化
    if hasattr(best_model, 'feature_importances_'):
        importances = best_model.feature_importances_
        feature_importance = pd.DataFrame({
            'Feature': features,
            'Importance': importances
        }).sort_values('Importance', ascending=False)

        plt.figure(figsize=(12, 8))
        sns.barplot(x='Importance', y='Feature', data=feature_importance)
        plt.title(f'{best_model_name} 特征重要性')
        plt.tight_layout()
        plt.show()

    # 12. 残差分析
    test_pred = results[best_model_name]['test_pred']
    residuals = y_test - test_pred

    # 残差分布
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    sns.histplot(residuals, bins=50, kde=True)
    plt.title('残差分布')
    plt.xlabel('残差')

    # 残差vs预测值
    plt.subplot(1, 2, 2)
    sns.scatterplot(x=test_pred, y=residuals, alpha=0.5)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.title('残差 vs 预测值')
    plt.xlabel('预测价格')
    plt.ylabel('残差')

    plt.tight_layout()
    plt.show()

    # 13. 时间维度误差分析
    test_df = test.copy()
    test_df['Predicted'] = test_pred
    test_df['Residual'] = residuals
    test_df['Absolute_Error'] = np.abs(residuals)

    # 按月误差分析
    monthly_error = test_df.groupby('Month')['Absolute_Error'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Month', y='Absolute_Error', data=monthly_error, marker='o')
    plt.title('月均预测误差分析')
    plt.xticks(range(1, 13))
    plt.xlabel('月份')
    plt.ylabel('平均绝对误差(MAE)')
    plt.grid(True)
    plt.show()

    # 14. 实际vs预测时间序列
    test_df = test_df.sort_values('Date')
    plt.figure(figsize=(14, 7))
    plt.plot(test_df['Date'], test_df['Avg_Price'], label='实际价格', linewidth=2)
    plt.plot(test_df['Date'], test_df['Predicted'], label='预测价格', linestyle='--')
    plt.fill_between(test_df['Date'],
                     test_df['Predicted'] - 30,
                     test_df['Predicted'] + 30,
                     alpha=0.2, color='orange')
    plt.title('实际价格 vs 预测价格 (2017年)')
    plt.xlabel('日期')
    plt.ylabel('价格')
    plt.legend()
    plt.grid(True)
    plt.show()

print("\n南瓜价格预测建模完成！")