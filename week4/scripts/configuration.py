# 全局配置、常量

import os
import tempfile

# ------------- 环境 -------------
os.environ['TMP']  = 'C:\\TEMP'
os.environ['TEMP'] = 'C:\\TEMP'
tempfile.tempdir   = 'C:\\TEMP'

# ------------- 路径 -------------
DATA_PATH      = 'data/US-pumpkins.csv'
OUTPUT_DIR     = 'output'

# ------------- 随机种子 -------------
RANDOM_STATE   = 42

# ------------- 模型搜索空间 -------------
GB_PARAM_GRID = {
    'n_estimators':      [100, 200],
    'learning_rate':     [0.05, 0.1],
    'max_depth':         [3, 5],
    'min_samples_split': [2, 5]
}

XGB_PARAM_GRID = {
    'n_estimators':  [100, 200],
    'learning_rate': [0.05, 0.1],
    'max_depth':     [3, 5],
    'subsample':     [0.8, 1.0]
}

# ------------- 绘图风格 -------------
import matplotlib.pyplot as plt
import seaborn as sns
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_theme(font="SimHei")