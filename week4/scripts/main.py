import warnings
warnings.filterwarnings('ignore')

from utility import log
from data_analysis import clean, engineer, load_raw
from feature_processing import build_dataset
from model import build_models
from evaluate import evaluate_model
import pandas as pd

def main():
    log('Step1: load & clean')
    df = engineer(clean(load_raw()))

    log('Step2: build dataset')
    X_train, y_train, X_test, y_test, features, scaler = build_dataset()

    # 为了在 evaluate 里画图，保留测试集 DataFrame
    test_df = df[df['Year'] == 2017][['Date','Avg_Price']].copy()

    log('Step3: train models')
    models = build_models()
    results = {}
    for name, mdl in models.items():
        mdl.fit(X_train, y_train)
        results[name] = evaluate_model(
            name, mdl.best_estimator_, X_train, y_train,
            X_test, y_test, features, test_df
        )

    # 选出最佳
    best = max(results.items(), key=lambda kv: kv[1]['test']['R2'])
    log(f'Best model: {best[0]} (R2={best[1]["test"]["R2"]:.4f})')

if __name__ == '__main__':
    main()