# 评估指标、可视化、残差分析

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from utility import log, save_csv, save_json

def metrics(y_true, y_pred):
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae  = mean_absolute_error(y_true, y_pred)
    r2   = r2_score(y_true, y_pred)
    return {'RMSE': rmse, 'MAE': mae, 'R2': r2}

def evaluate_model(name, model, X_train, y_train, X_test, y_test, features, test_df):
    """对单个模型评估并画图"""
    log(f'Evaluating {name} ...')
    train_pred = model.predict(X_train)
    test_pred  = model.predict(X_test)

    res = {
        'train': metrics(y_train, train_pred),
        'test' : metrics(y_test,  test_pred),
        'pred' : test_pred
    }

    # 特征重要性
    if hasattr(model, 'feature_importances_'):
        fi = pd.DataFrame({'Feature': features, 'Importance': model.feature_importances_}) \
               .sort_values('Importance', ascending=False)
        plt.figure(figsize=(10,6))
        sns.barplot(x='Importance', y='Feature', data=fi)
        plt.title(f'{name} Feature Importance')
        plt.tight_layout()
        plt.savefig(f'output/{name}_feature_importance.png')
        plt.close()

    # 残差
    residuals = y_test - test_pred
    plt.figure(figsize=(12,5))
    plt.subplot(1,2,1)
    sns.histplot(residuals, bins=50, kde=True)
    plt.title('Residual Distribution')

    plt.subplot(1,2,2)
    sns.scatterplot(x=test_pred, y=residuals, alpha=0.5)
    plt.axhline(0, ls='--', c='r')
    plt.title('Residual vs Predicted')
    plt.tight_layout()
    plt.savefig(f'output/{name}_residual.png')
    plt.close()

    # 时间序列
    test_df = test_df.assign(Predicted=test_pred, Residual=residuals).sort_values('Date')
    plt.figure(figsize=(14,6))
    plt.plot(test_df['Date'], test_df['Avg_Price'], label='Actual')
    plt.plot(test_df['Date'], test_df['Predicted'], label='Predicted', ls='--')
    plt.fill_between(test_df['Date'],
                     test_df['Predicted']-30,
                     test_df['Predicted']+30,
                     alpha=0.2, color='orange')
    plt.title(f'{name}: Actual vs Predicted (2017)')
    plt.legend(); plt.grid(True)
    plt.savefig(f'output/{name}_series.png')
    plt.close()

    save_json(res, f'{name}_metrics')
    save_csv(test_df[['Date','Avg_Price','Predicted','Residual']], f'{name}_pred.csv')
    return res