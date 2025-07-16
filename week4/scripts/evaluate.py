# 评估指标、可视化、残差分析

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'KaiTi', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False
plt.rcParams['font.size'] = 10
sns.set_theme(font="SimHei")  # 设置seaborn全局字体


def evaluate_model(model, X_train, y_train, X_test, y_test):
    # 预测
    train_pred = model.predict(X_train)
    test_pred = model.predict(X_test)

    # 计算指标
    train_rmse = np.sqrt(mean_squared_error(y_train, train_pred))
    test_rmse = np.sqrt(mean_squared_error(y_test, test_pred))
    test_mae = mean_absolute_error(y_test, test_pred)
    test_r2 = r2_score(y_test, test_pred)

    return {
        'train_pred': train_pred,
        'test_pred': test_pred,
        'train_rmse': train_rmse,
        'test_rmse': test_rmse,
        'test_mae': test_mae,
        'test_r2': test_r2
    }


def visualize_results(model, results, test_df, features):
    # 特征重要性
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

    # 残差分析
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

    # 时间维度误差分析
    test_df = test_df.copy()
    test_df['Predicted'] = test_pred
    test_df['Residual'] = residuals
    test_df['Absolute_Error'] = np.abs(residuals)

    monthly_error = test_df.groupby('Month')['Absolute_Error'].mean().reset_index()

    plt.figure(figsize=(10, 6))
    sns.lineplot(x='Month', y='Absolute_Error', data=monthly_error, marker='o')
    plt.title('月均预测误差分析')
    plt.xticks(range(1, 13))
    plt.xlabel('月份')
    plt.ylabel('平均绝对误差(MAE)')
    plt.grid(True)
    plt.show()

    # 实际vs预测时间序列
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

if __name__ == '__main__':
    import joblib
    import os
    from data_analysis import load_and_clean_data, prepare_features
    from feature_processing import scale_features

    # 加载模型
    model = joblib.load('../output/lightgbm_best.pkl')

    # 数据准备
    file_path = os.path.abspath(r'D:\Python\pycharm\xiaoxueqi\week4\data\US-pumpkins.csv')
    df = load_and_clean_data(file_path)
    X_train, X_test, y_train, y_test = prepare_features(df)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)
    test_df = df[df['Year'] == 2017].copy()

    # 评估
    from evaluate import evaluate_model, visualize_results
    results = evaluate_model(model, X_train_scaled, y_train, X_test_scaled, y_test)
    visualize_results(model, results, test_df, X_train.columns.tolist())