import os
from data_analysis import load_and_clean_data, prepare_features
from feature_processing import scale_features
from model import train_model
from evaluate import evaluate_model, visualize_results
from utility import setup_visualization, log_model_performance, save_results
from configuration import conf


def main():
    # 初始化可视化设置
    setup_visualization()

    # 数据准备
    df = load_and_clean_data(r'D:\Python\pycharm\xiaoxueqi\week4\data\US-pumpkins.csv')
    X_train, X_test, y_train, y_test = prepare_features(df)
    test_df = df[df['Year'] == 2017].copy()

    # 特征缩放
    X_train_scaled, X_test_scaled, _ = scale_features(X_train, X_test)

    # 训练和评估
    all_results = {}
    models = {}

    for model_name in conf.keys():
        print(f"\n训练 {model_name} 模型...")
        from model import train_model_optuna
        best_model, best_params = train_model_optuna(
            model_name, X_train_scaled, y_train, n_trials=40
        )
        print(f"{model_name} Optuna 最佳参数：{best_params}")


        # 评估模型
        model_results = evaluate_model(best_model, X_train_scaled, y_train, X_test_scaled, y_test)

        # 记录结果
        log_model_performance(model_name, model_results)
        models[model_name] = best_model
        all_results[model_name] = {
            'params': best_params.best_params_,
            'metrics': {
                'train_rmse': model_results['train_rmse'],
                'test_rmse': model_results['test_rmse'],
                'test_mae': model_results['test_mae'],
                'test_r2': model_results['test_r2']
            }
        }

    # 保存结果
    output_dir = os.path.join(os.path.dirname(__file__), '..', 'output')
    os.makedirs(output_dir, exist_ok=True)
    save_results(all_results, os.path.join(output_dir, 'model_results.json'))

    # 选择最佳模型
    best_model_name = max(all_results,
                          key=lambda x: all_results[x]['metrics']['test_r2'])
    best_model = models[best_model_name]
    print(f"\n最佳模型: {best_model_name} (R²={all_results[best_model_name]['metrics']['test_r2']:.4f})")

    # 重新预测用于可视化
    y_train_pred = best_model.predict(X_train_scaled)
    y_test_pred = best_model.predict(X_test_scaled)

    viz_results = {
        'train_pred': y_train_pred,
        'test_pred': y_test_pred,
        **all_results[best_model_name]['metrics']
    }

    visualize_results(best_model, viz_results, test_df, X_train.columns.tolist())
    print("\n南瓜价格预测建模完成！")


if __name__ == '__main__':
    main()