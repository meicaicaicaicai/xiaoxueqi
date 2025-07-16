# 模型定义、超参搜索、训练

from sklearn.model_selection import GridSearchCV
from configuration import conf, tscv


def train_model(model_name, X_train, y_train):
    model_config = conf[model_name]

    # 初始化模型
    model = model_config["model_name"](**model_config["model_params"])

    # 网格搜索
    grid_search = GridSearchCV(
        model,
        param_grid=model_config["param_grid"],
        cv=tscv,
        scoring='neg_mean_squared_error',
        n_jobs=-1,
        verbose=1
    )

    # 训练模型
    grid_search.fit(X_train, y_train)

    return grid_search

if __name__ == '__main__':
    import joblib
    from data_analysis import load_and_clean_data, prepare_features
    from feature_processing import scale_features
    import os

    # 数据准备
    file_path = os.path.abspath(r'D:\Python\pycharm\xiaoxueqi\week4\data\US-pumpkins.csv')
    df = load_and_clean_data(file_path)
    X_train, X_test, y_train, y_test = prepare_features(df)
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 训练 LightGBM
    from configuration import conf
    grid = train_model('LightGBM', X_train_scaled, y_train)
    print('最佳参数:', grid.best_params_)
    print('CV 最佳分数:', -grid.best_score_)

    # 保存模型
    os.makedirs('../output', exist_ok=True)
    joblib.dump(grid.best_estimator_, '../output/lightgbm_best.pkl')