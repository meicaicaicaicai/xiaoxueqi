# 通用工具函数

import seaborn as sns
import os
import json
import pandas as pd
import matplotlib.pyplot as plt

def setup_visualization():
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']
    plt.rcParams['axes.unicode_minus'] = False
    plt.rcParams['font.size'] = 10
    sns.set_theme(font="SimHei")

def save_results(results, filename):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    with open(filename, 'w') as f:
        json.dump(results, f, indent=4)

def log_model_performance(model_name, results):
    print(f"{model_name} 性能:")
    print(f"  训练RMSE: {results['train_rmse']:.2f}")
    print(f"  测试RMSE: {results['test_rmse']:.2f}")
    print(f"  测试MAE: {results['test_mae']:.2f}")
    print(f"  测试R²: {results['test_r2']:.4f}")