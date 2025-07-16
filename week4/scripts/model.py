# 模型定义、超参搜索、训练










# model_optuna.py
import optuna
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from configuration import conf, tscv   # tscv 仍可用
import numpy as np

def train_model_optuna(model_name, X_train, y_train, n_trials=30):
    cfg = conf[model_name]
    ModelCls = cfg["model_name"]
    base_params = cfg["model_params"]

    def objective(trial):
        # 1. 定义搜索空间
        params = {
            **base_params,
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'max_depth': trial.suggest_int('max_depth', 3, 12),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0)
        }

        # 2. 建模型
        model = ModelCls(**params)

        # 3. 交叉验证评估（负 MSE，Optuna 默认求最小值）
        neg_mse = cross_val_score(
            model, X_train, y_train,
            cv=tscv,
            scoring='neg_mean_squared_error',
            n_jobs=-1
        ).mean()
        return -neg_mse     # Optuna 越小越好，所以取反

    # 4. 启动搜索
    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=n_trials, show_progress_bar=True)

    # 5. 返回最佳模型
    best_params = {**base_params, **study.best_params}
    best_model = ModelCls(**best_params)
    best_model.fit(X_train, y_train)
    return best_model, study.best_params

if __name__ == '__main__':
    import joblib
    import os
    from data_analysis import load_and_clean_data, prepare_features
    from feature_processing import scale_features
    from model import train_model_optuna

    # 1. 数据准备
    file_path = os.path.abspath(r'D:\Python\pycharm\xiaoxueqi\week4\data\US-pumpkins.csv')
    df = load_and_clean_data(file_path)
    X_train, X_test, y_train, y_test = prepare_features(df)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 2. 训练 LightGBM（Optuna）
    best_model, best_params = train_model_optuna(
        model_name='LightGBM',
        X_train=X_train_scaled,
        y_train=y_train,
        n_trials=20
    )
    print('Optuna 最佳参数:', best_params)


    # 4. 保存模型
    os.makedirs('../output', exist_ok=True)
    joblib.dump(best_model, '../output/lightgbm_best.pkl')
    print('模型已保存到 ../output/lightgbm_best.pkl')