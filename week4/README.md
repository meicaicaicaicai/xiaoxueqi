# data_analysis.py
## 函数load_and_clean_data(file_path)
### 配置了系统环境变量和Python的tempfile模块后，设置环境变量TMP和TEMP的值为C:\TEMP，同时设置tempfile模块的临时目录为C:\TEMP
```python
import os  
import tempfile  
os.environ['TMP'] = 'C:\\TEMP'  
os.environ['TEMP'] = 'C:\\TEMP'  
tempfile.tempdir = 'C:\\TEMP'
```
1. os 模块提供了与操作系统交互的功能，包括设置和获取环境变量；tempfile 模块用于创建临时文件和目录
2. os.environ是个字典，用于访问和设置环境变量；分别设置环境变量TMP和TEMP的值为C:\TEMP
3. tempfile.tempdir表示设置 tempfile 模块的临时目录属性，这会告诉 tempfile模块在创建临时文件和目录时使用指定的路径C:\TEMP

### 加载数据，解决读取文件出现乱码问题，把第一列名字命名为City Name
```python
df = pd.read_csv(file_path, encoding='latin1')
# 把第一列强行命名为 City Name
df = df.rename(columns={df.columns[0]: 'City Name'})
```
1. encoding='latin1'：指定了读取文件时使用的字符编码为latin1，用于正确解码西欧语言字符  
2. 如果文件来自非西欧语言环境（中文），优先尝试 encoding='utf-8' 或 encoding='gbk'
3. df.columns获取DataFrame的所有列名,返回一个索引对象,[0]表示拉取第一个列名
4. {df.columns[0]: 'City Name'}表示创建一个字典，指定旧列名到新列名的映射关系。一般格式为{原始列名: 新列名}
5. .rename(columns=...)是pandas的DataFrame方法，用于重命名列，而columns 参数接受字典，指定要修改的列名映射
6. df = 会覆盖原始DataFrame
 
### 移除空值
```python
empty_cols = [col for col in df.columns if df[col].isnull().all()]
df = df.drop(columns=empty_cols)
```
1. df.columns表示获取数据表 df 的所有列名；  
df[col].isnull()：检查列 col 中的每个值是否为 NaN，返回一个布尔值序列；  
.all()表示所有；  
[col for col in df.columns if df[col].isnull().all()]这是一个列表推导式，用于筛选出所有完全为空的列名，并将这些列名存储在 empty_cols 列表中
2. df.drop(columns=empty_cols)表示删除数据表 df 中指定的列，即 empty_cols 列表中的列


### 识别并删除数据表df中低方差，即唯一值数量非常少的列。具体来说，它会删除那些唯一值数量小于或等于1的列，而这些列通常对模型的训练和预测没有帮助
```python
low_variance_cols = [col for col in df.columns if df[col].nunique() <= 1]
df = df.drop(columns=low_variance_cols)
 ```   
1. df.columns;  
df[col].nunique()表示计算列 col 中的唯一值数量；  
[col for col in df.columns if df[col].nunique() <= 1]：这是一个列表推导式，用于筛选出那些唯一值数量小于或等于 1 的列名，并将这些列名存储在low_variance_cols列表中
2. df.drop(columns=low_variance_cols)：删除数据表 df 中指定的列，即 low_variance_cols 列表中的列

### 选择关键特征
```python
selected_cols = ['City Name', 'Package', 'Variety', 'Item Size', 'Color',
                     'Origin', 'Date', 'Low Price', 'High Price']  
df = df[selected_cols]
```
1. 定义了一个名为selected_cols的列表，放入了若干个字符串
2. 将数据表df重新筛选为只包含selected_cols列表中的列

### 对数据表中的日期列进行处理，确保日期格式正确，并去除无法解析为日期的行
```python
df['Date'] = pd.to_datetime(df['Date'], errors='coerce', format='%m/%d/%y')  
df = df[~df['Date'].isnull()]
```
1. pd.to_datetime()是pandas库中用于将字符串转换成日期时间格式的函数
2. df['Date']表示是数据表中名为Date的列
3. errors='coerce'是个参数，作用为如果某个值无法解析为日期，则将其转换成NaT（NaT表示非时间值）
4. format='%m%d%y'也是个参数，作用是指定了日期的格式，表示日期字符串的格式是“月/日/年”

### 从 'Date' 列中提取日期的年份、月份和周数，并将它们分别存储为新的列  
```python
df['Year'] = df['Date'].dt.year  
df['Month'] = df['Date'].dt.month  
df['Week'] = df['Date'].dt.isocalendar().week
```
1. .dt是pandas提供的一个访问器，用于操作日期时间类型的列
2. .year是.dt访问器的一个属性，用于提取日期中的年份；.month提取月份
3. .isocalendar()是.dt访问器的一个方法，用于将日期转换成日历格式，返回一个包含年，周和星期几的元组，  
.week是从.isocalendar()返回的元组中提取周数的部分

### 根据月份信息为数据表添加两个新的列：'Season' 和 'Is_Holiday'  
```python
df['Season'] = df['Month'].apply(lambda x: 1 if x in [9, 10, 11] else  
2 if x in [12, 1, 2] else  
3 if x in [3, 4, 5] else 4)  
# 10、11月万圣节，需求高峰  
df['Is_Holiday'] = df['Month'].isin([10, 11]).astype(int)
```
1. .apply(lambda x:...)是pandas的一个方法，用于对列中的每个值应用一个函数。  
这里使用了lambda匿名函数来定义一个简单的条件逻辑
2. lambda x:1 if x in [9,10,11] else ... 表示如果月份x是9,10,11，则返回1（秋季）  
2 if x in [12,1,2] else ... 表示如果月份x在12,1,2月份则返回2（冬季）  
3 if x in [3,4,5] else 4 表示如果月份x在3,4,5月份则返回3（春季），否则返回4（夏季）
3. df['Month'].isin([10,11])表示检查Month列中值是否存在10和11，结果返回一个布尔值序列
4. .astype(int) 表示将布尔值序列转换成整数，True换成1，False换成0

### 计算数据表中每行的平均价格，并将结果存储到一个新的列 'Avg_Price' 中  
```python
df['Avg_Price'] = (df['Low Price'] + df['High Price']) / 2
```

### 处理数据表中某些列的缺失值（NaN），将这些缺失值替换为字符串 'UNKNOWN'
```python
df['Color'].fillna('UNKNOWN', inplace=True)
df['Item Size'].fillna('UNKNOWN', inplace=True)
df['Origin'].fillna('UNKNOWN', inplace=True)
```
1. .fillna('UNKNOWN')是pandas中的一个方法，用于填充列中的缺失值NaN，这里把缺失值换成了UNKNOWN
2. inplace=Ture 参数表示是直接在数据表df上面改，不是返回一个新的数据表

### 保留平均价格在 0 到 1000 之间的行,保留年份大于或等于 2016 的行
```python
df = df[(df['Avg_Price'] > 0) & (df['Avg_Price'] < 1000)]
df = df[df['Year'] >= 2016]
```
1. &是逻辑与操作符，表示两个条件必须同时满足
2. df[...] 是pandas的布尔索引功能，用于根据布尔序列筛选数据
3. 选取年份大于2016是因为较新的数据更能反映当前的市场状况和消费者行为，而较早的数据可能不再适用于当前的分析，市场条件可能已经发生了显著变化

### 先对数据表进行排序，然后计算每个产地'Origin'的前一周价格'Prev_Week_Price'以及价格的变化率'Price_Change'
```python
df = df.sort_values(['Origin', 'Date'])
df['Prev_Week_Price'] = df.groupby('Origin')['Avg_Price'].shift(1)  
df['Price_Change'] = df.groupby('Origin')['Avg_Price'].pct_change()
```
1. df.sort_values(['A','B'])是pandas的排序方法，首先按A排序，对于同一个类型的数据再进行B排序  
排序后，数据表中的行会按照产地分组，并且每个产地内数据会按照日期顺序排列
2. df.groupby('A')是pandas的分组方法，即按照A列的值将数据表进行分组  
['Avg_Price']表示对完成分组后的Avg_Price列进行操作（制定列）  
.shift(1)是pandas一个方法，用于将列中的值向下移动一行。这样子，对于每个产地，当前行的Prev_Week_Price就是前一周的Avg_Price
3. .pct_change()是pandas一个方法，用于计算列中相邻值的百分比变化率。即为每个产地的每一行计算平均价格的变化率，并将结果存储在新列 'Price_Change' 中

### 计算每个产地'Origin'的平均价格，并将这个整体价格水平'Region_Avg_Price'合并回原始数据表df中
```python
region_avg = df.groupby('Origin')['Avg_Price'].mean().reset_index()  
region_avg.columns = ['Origin', 'Region_Avg_Price']  
df = pd.merge(df, region_avg, on='Origin', how='left')
```
1. .mean()表示计算每个分组的平均价格
2. .reset_index()表示将分组后的结果转化为一个新的数据表，其中包含两列Origin和Avg_Price
3. .columns表示将新数据表region_avg中列名重新命名为Origin和Region_Avg_Price
4. .merge(A,B)表示将A与B合并，即将旧的数据表df和新生成的数据表region_avg合并，其中  
on='Origin'指定合并的键是Origin列，how='left'表示左连接

### 统计每个品种（'Variety'）的出现次数，并将这个统计结果合并回原始数据表df中，同时添加一个标记列'Rare_Variety'，用于标识稀有品种***   
```python
variety_count = df['Variety'].value_counts().reset_index()      
variety_count.columns = ['Variety', 'Variety_Count']    
df = pd.merge(df, variety_count, on='Variety', how='left')   
df['Rare_Variety'] = (df['Variety_Count'] < 100).astype(int)
```
1. df['Varity'].value_counts()表示计算'Variety'列中的值出现次数，然后返回一个Series（数组），其中索引是variety,值是出现次数
2. .reset_index()用于将这个Series转换为一个DataFrame,其中包含两列：索引列和值列
3. .columns重新命名
4. .merge()合并
5. .astype(int)用于将布尔值序列转换为整数

### 从数据表df中删除那些在指定列中包含缺失值的行，返回结果
```python
df.dropna(subset=['Prev_Week_Price', 'Price_Change'], inplace=True)  
return df
```
1. df.dropna()是pandas中一个方法，用于删除数据表中包含缺失值的行
2. subset=[A,B]表示指定只检查A,B这两列
3. inplace=True

## 函数prepare_features(df)
### 将数据表df中的分类变量转换为数值型变量
```python
cat_cols = ['City Name', 'Package', 'Variety', 'Item Size', 'Color', 'Origin']
for col in cat_cols:
    le = LabelEncoder()
    df[col] = le.fit_transform(df[col].astype(str))
```
1. LabelEncoder是sklearn.preprocessing模块中的一个类，用于将分类变量转换为数值型变量
创建一个LabelEncoder对象，即le
2. .fit_transform方法会将每个唯一的分类值映射到一个唯一的整数值
3. .astype(str)?

### 将数据集分割为2016年为训练集，2017年为测试集；提取训练集和测试集的特征和目标变量
```python
train = df[df['Year'] == 2016]
test = df[df['Year'] == 2017]

X_train = train[features]
y_train = train['Avg_Price']
X_test = test[features]
y_test = test['Avg_Price']

return X_train, X_test, y_train, y_test
```
2016年为训练集，2017年为测试集

# model.py
## 函数train_model_optuna(model_name, X_train, y_train, n_trials=30)
### 调取所需要的模型配置到一个字典，接着调取其字典中模型名称与参数到不同变量与字典
```python
cfg = conf[model_name]
ModelCls = cfg["model_name"]
base_params = cfg["model_params"]
```
1. 调模型配置：  
conf是一个字典，有很多不同模型的配置信息；  
model_name是一个变量，表示当前所需要的模型的名称；  
conf[model_name]表示从conf中提取键为model_name的值
2. 调模型类名：  
从cfg中提取键为model_name的值（字符串）
3. 从cfg中提取键为model_params的值（字典），这个值是模型基础参数

### 函数objective(trial)
#### 建立包含基础参数与超参数的字典
```python
params = {
    **base_params,
    'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
    'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
    'max_depth': trial.suggest_int('max_depth', 3, 12),
    'subsample': trial.suggest_float('subsample', 0.6, 1.0)
}
```
1. **base_params表示将base_params中的所有基础参数合并到params字典中
2. 用到了Optuna，这是一个用于超参数优化的库，它通过自动搜索超参数来找到最优的模型性能。  
3. trial.suggest_int是Optuna的一个方法，用于在指定范围内生成一个整数值。n_estimators是超参数名称，表示模型中树的数量。范围是100到1000  
4. trial.suggest_float是Optuna的一个方法，用于在指定范围内生成一个浮点值。learing_rate是超参数名称，表示学习率，范围0.01到0.3。log=True表示使用对数空间搜索。
5. max_depth是超参数名称，表示树的最大深度，范围3到12
6. subsample是超参数名称，表示子采样率，控制用于训练每棵树的数据比例。范围0.6到1.0

#### 建立模型
```python
model = ModelCls(**params)
```
**params表示将params字典传递给模型类（ModelCls）的构造函数，实例化模型

#### 定义了使用时间序列交叉验证来评估模型性能并返回正值
```python
neg_mse = cross_val_score(
    model, X_train, y_train,
    cv=tscv,
    scoring='neg_mean_squared_error',
    n_jobs=-1
).mean()
return -neg_mse 
```
1. cross_val_score表示使用交叉验证来评估模型性能，其中：  
model表示待评估的模型；X_train表示训练集特征；y_train表示训练集的目标变量；  
cv=tscv表示指定交叉验证策略，即tscv，时间序列交叉验证；  
scoring='neg_mean_squared_error'表示使用负均方误差作为评分标准；  
n_jobs=-1表示使用所有可用的CPU核心进行并行运算，这样子可以显著提高计算效率，减少训练时间，充分利用硬件资源
2. .mean()表示计算交叉验证分数的均值
3. Optuna默认寻找最小值，而cross_val_score返回的是负均方误差，因此要取个符号使其返回正值

### 创建优化研究对象并运行优化进程，找到最优模型性能
```python
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=n_trials, show_progress_bar=True)
```
1. .creat_study()表示创建一个Optuna的优化研究对象；  
direction='minimize'表示指定优化方向为最小化目标函数，说明Optuna会尝试找到使目标函数值最小的超参数组合
2. study.optimize方法，来进行优化过程，其中：  
objective函数定义了如何评估模型的性能，并返回一个需要优化的值；  
n_trial=100表示Optuna将运行100次目标函数，每次都尝试不同的超参数组合；  
通过show_progress_bar=True来显示进度条

### 合并基础参数与优化得到的超参数后，使用这些参数实例化并训练最佳模型，最后返回训练好的最佳模型和最佳超参数
```python
best_params = {**base_params, **study.best_params}
best_model = ModelCls(**best_params)
best_model.fit(X_train, y_train)
return best_model, study.best_params
```
1. {**A,**B}，其中**表示将两个字典合并为一个字典。如果两个字典有相同键，A中值会覆盖B中值
2. ModelCls指模型类引用
3. best_model.fit(X_train,y_train)表示使用训练数据X_train和目标变量y_train训练最佳模型
4. study.best_params为最佳超参数

# evaluate.py
## 函数evaluate_model(model, X_train, y_train, X_test, y_test)
### 使用训练好的模型对训练集与测试集进行预测
```python
train_pred = model.predict(X_train)
test_pred = model.predict(X_test)
```
1. model表示已经训练好的模型；X_train表示训练集的特征数据；.predict()方法表示对训练集的特征数据进行预测
2. X_test为测试集的特征数据

### 计算模型在训练集上的均方根误差（RMSE）。在测试集上的均方根误差（RMSE），平均绝对误差（MAE），决定系数（R²）。
```python
train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
test_mae = mean_absolute_error(y_test, test_pred)
test_r2 = r2_score(y_test, test_pred)
```
1. mean_squared_error()表示计算训练集（y_train）的真实值y_train和预测值train_pred之间的均方误差（MSE）
2. np.sqrt()表示对均方误差取平方根，得到均方根误差（RMSE）
3. mean_absolute_error(y_test,test_pred)表示计算测试集的真实值y_test和预测值test_pred之间的平均绝对误差（MAE)
4. r2+score(y_test,test_pred)表示计算测试集的真实值y_test和预测值test_pred之间的决定系数（R²）

### 返回值
```python
return {
    'train_pred': train_pred,
    'test_pred': test_pred,
    'train_rmse': train_rmse,
    'test_rmse': test_rmse,
    'test_mae': test_mae,
    'test_r2': test_r2
}
```

## 函数visualize_results(model, results, test_df, features)
### 检查模型是否含有（if)特征重要性属性，接着提取特征重要性，然后创建一个包含特征名称和特征重要性的DataFrame，最后使用条形图来可视化
```python
if hasattr(model, 'feature_importances_'):
    importances = model.feature_importances_
    feature_importance = pd.DataFrame({
        'Feature': features,
        'Importance': importances
    }).sort_values('Importance', ascending=False)

    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title(f'{type(model).__name__} 特征重要性')
    plt.tight_layout()
    plt.show()
```
1. hasattr(model, 'feature_importances_')表示检查模型是否具有feature_importances_属性
2. model.feature_importances_表示获取模型的特征重要性值，通常是一个数组，长度与特征数量相同
3. features为特征列的名称列表；importances为特征重要性值；  
sort_values('Importance',ascending=False)表示按特征重要性降序排列
4. plt.figure(figsize=(12, 8))：设置图形的大小为 12x8 英寸；  
sns.barplot(x='Importance', y='Feature', data=feature_importance)：使用 Seaborn 的 barplot 绘制条形图，x 轴表示特征重要性，y 轴表示特征名称，数据为feature_importance;  
plt.title(f'{type(model).__name__} 特征重要性')：设置图形的标题，显示模型的类名;  
plt.tight_layout()：调整布局，确保标签和标题不会被截断

### 通过绘制残差分布图和残差与预测值的关系图来直观的评估预测误差
```python
test_pred = results['test_pred']
residuals = test_df['Avg_Price'] - test_pred

plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
sns.histplot(residuals, bins=50, kde=True)
plt.title('残差分布')
plt.xlabel('残差')

plt.subplot(1, 2, 2)
sns.scatterplot(x=test_pred, y=residuals, alpha=0.5)
plt.axhline(y=0, color='r', linestyle='--')
plt.title('残差 vs 预测值')
plt.xlabel('预测价格')
plt.ylabel('残差')
plt.tight_layout()
plt.show()
```
1. test_df['Avg_Price']为测试集的真实目标值
2. plt.subplot(1, 2, 1)表示创建一个 1x2 的子图网格，并选择第一个子图；  
sns.histplot(residuals, bins=50, kde=True)：使用 Seaborn 的 histplot 绘制残差的直方图，包含 50 个柱子，并添加核密度估计（KDE）曲线
3. sns.scatterplot(x=test_pred, y=residuals, alpha=0.5)：使用 Seaborn 的 scatterplot 绘制预测值与残差的散点图，点的透明度为 0.5；  
plt.axhline(y=0, color='r', linestyle='--')：在 y=0 处绘制一条红色虚线，表示残差为 0 的位置； 

### 在测试集副本中添加预测值，残差和绝对误差列，接着按月份计算平均绝对误差，绘制月均预测误差折线图
```python
test_df = test_df.copy()
test_df['Predicted'] = test_pred
test_df['Residual'] = residuals
test_df['Absolute_Error'] = np.abs(residuals)

# 计算每月的平均绝对误差
monthly_error = test_df.groupby('Month')['Absolute_Error'].mean().reset_index()

# 绘制月均预测误差的折线图
plt.figure(figsize=(10, 6))
sns.lineplot(x='Month', y='Absolute_Error', data=monthly_error, marker='o')
plt.title('月均预测误差分析')
plt.xticks(range(1, 13))
plt.xlabel('月份')
plt.ylabel('平均绝对误差(MAE)')
plt.grid(True)
plt.show()
```
1. test_df.copy()表示创建测试集副本，避免直接修改原始数据；  
依次添加预测值列，残差列，绝对误差列;  
np.abs() 是 NumPy 库中的一个函数，用于计算输入值的绝对值:    
numpy.abs(x, /, out=None, *, where=True, casting='same_kind', order='K', dtype=None, subok=True[, signature, extobj]) = <ufunc 'absolute'>  
x：输入值，可以是单个数字、数组或其他可迭代对象。  
out（可选）：输出数组，用于存储结果。  
where（可选）：布尔数组，指定哪些位置的值需要计算。  
casting（可选）：指定类型转换的规则。  
order（可选）：指定输出数组的内存布局。  
dtype（可选）：指定输出数组的数据类型。  
subok（可选）：如果为 True，则子类将被传递.
2. test_df.groupby('Month')表示按月份对测试集进行分组；  
['Absolute_Error'].mean()表示计算每个分组（即每个月）的平均绝对误差；  
3. .reset_index()表示将结果转换为 DataFrame，其中包含两列：'Month' 和 'Absolute_Error'
4. sns.lineplot(x='Month', y='Absolute_Error', data=monthly_error, marker='o')：使用 Seaborn 的 lineplot 绘制折线图，x 轴表示月份，y 轴表示平均绝对误差，使用圆圈标记每个数据点；  
plt.xticks(range(1, 13))：设置 x 轴的刻度范围为 1 到 12，表示 12 个月份；  
plt.grid(True)表示添加网格线

### 按照日期对测试集进行排序后，绘制实际价格和预测价格的对比图，接着添加预测价格的置信区间，添加标题，标签，图例和网格线
```python
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
```
1. test_df.sort_values('Date')表示按 'Date' 列对测试集进行排序，确保数据按时间顺序排列
2. plt.plot(test_df['Date'], test_df['Avg_Price'], label='实际价格', linewidth=2)：绘制实际价格的折线图，使用实线，线宽为2；  
plt.plot(test_df['Date'], test_df['Predicted'], label='预测价格', linestyle='--')：绘制预测价格的折线图，使用虚线
3. plt.fill_between(test_df['Date'], test_df['Predicted'] - 30, test_df['Predicted'] + 30, alpha=0.2, color='orange')：在预测价格上下各加减 30 的范围内填充颜色，表示置信区间；  
alpha=0.2 表示填充颜色的透明度为 20%，color='orange' 表示填充颜色为橙色
4. plt.legend()：添加图例；plt.grid(True)：添加网格线