@@ -6,7 +6,6 @@ from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
from statsmodels.tsa.seasonal import seasonal_decompose
import matplotlib as mpl

# ========== 字体设置 - 解决中文显示问题 ==========
try:
@@ -83,31 +82,59 @@ df_cleaned['Price_per_cuft'] = df_cleaned['Avg_Price'] / df_cleaned['Volume']

print("\n清洗后数据形状:", df_cleaned.shape)

# ========== 单独显示每个图表 ==========
# ========== 可视化分析 - 不保存图片 ==========

# 1. 品种对价格的影响
plt.figure(figsize=(10, 6))
top_varieties = df_cleaned['Variety'].value_counts().nlargest(5).index
variety_subset = df_cleaned[df_cleaned['Variety'].isin(top_varieties)]
variety_prices = variety_subset.groupby('Variety')['Price_per_cuft'].median().sort_values()
variety_prices.plot(kind='bar', color='darkorange')
plt.title('不同南瓜品种的价格比较(每立方英尺)')
plt.ylabel('价格 ($/cuft)')
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
plt.savefig('variety_price_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 2. 大小对价格的影响
plt.figure(figsize=(10, 6))
size_order = ['small', 'medium', 'med-large', 'large', 'xlarge', 'jumbo', 'exjumbo']

# 动态获取大小顺序
size_order = df_cleaned['Item Size'].value_counts().index

# 计算价格中位数
size_prices = df_cleaned.groupby('Item Size')['Price_per_cuft'].median().reindex(size_order)
size_prices.plot(kind='bar', color='sandybrown')
plt.title('南瓜大小对价格的影响')
plt.ylabel('价格 ($/cuft)')

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
plt.savefig('size_price_impact.png', dpi=300, bbox_inches='tight')
plt.show()

# 3. 时间序列分析（月度）
@@ -120,7 +147,6 @@ plt.xlabel('月份')
plt.xticks(range(1, 13), ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec'])
plt.grid(True)
plt.tight_layout()
plt.savefig('monthly_price_trend.png', dpi=300, bbox_inches='tight')
plt.show()

# 4. 季节性分解
@@ -132,7 +158,6 @@ plt.title('南瓜价格长期趋势')
plt.ylabel('价格 ($/cuft)')
plt.grid(True)
plt.tight_layout()
plt.savefig('price_long_term_trend.png', dpi=300, bbox_inches='tight')
plt.show()

# 5. 本地vs外地价格比较
@@ -144,7 +169,6 @@ plt.ylabel('平均价格 ($/cuft)')
plt.xticks([0, 1], ['外地', '本地'], rotation=0)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('local_vs_nonlocal_price.png', dpi=300, bbox_inches='tight')
plt.show()

# 6. 包装类型价格分布
@@ -155,7 +179,6 @@ plt.title('不同包装类型的价格比较')
plt.xlabel('价格 ($/cuft)')
plt.grid(axis='x', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('package_price_comparison.png', dpi=300, bbox_inches='tight')
plt.show()

# 7. 品种-大小交互热图
@@ -174,7 +197,6 @@ plt.ylabel('品种')
plt.xticks(range(len(pivot_data.columns)), pivot_data.columns, rotation=45)
plt.yticks(range(len(pivot_data.index)), pivot_data.index)
plt.tight_layout()
plt.savefig('variety_size_heatmap.png', dpi=300, bbox_inches='tight')
plt.show()

# ========== 预测模型 ==========
@@ -206,6 +228,13 @@ print(f"\n模型性能评估:")
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
@@ -218,7 +247,6 @@ plt.xlabel('特征重要性')
plt.title('南瓜价格预测特征重要性')
plt.gca().invert_yaxis()
plt.tight_layout()
plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
plt.show()

# 预测与实际值比较
@@ -230,13 +258,4 @@ plt.ylabel('预测价格')
plt.title('实际价格 vs 预测价格')
plt.grid(True)
plt.tight_layout()
plt.savefig('actual_vs_predicted.png', dpi=300, bbox_inches='tight')
plt.show()

# 查看训练集表现
y_train_pred = model.predict(X_train)
train_rmse = np.sqrt(mean_squared_error(y_train, y_train_pred))
train_r2 = r2_score(y_train, y_train_pred)

print(f"训练集表现: RMSE={train_rmse:.2f}, R²={train_r2:.2f}")
print(f"测试集表现: RMSE={rmse:.2f}, R²={r2:.2f}")
\ No newline at end of file