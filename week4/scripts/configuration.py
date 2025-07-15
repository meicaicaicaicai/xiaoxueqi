# 全局配置、常量

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

conf = {
    "GradientBoosting": {
        "model_name": GradientBoostingRegressor,
        "model_params": {
            "random_state": 42
        },
        "param_grid": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'min_samples_split': [2, 5]
        }
    },
    "XGBoost": {
        "model_name": XGBRegressor,
        "model_params": {
            "objective": 'reg:squarederror',
            "random_state": 42,
            "n_jobs": -1
        },
        "param_grid": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
    },
    "LightGBM": {
        "model_name": LGBMRegressor,
        "model_params": {
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        },
        "param_grid": {
            'n_estimators': [100, 200],
            'learning_rate': [0.05, 0.1],
            'max_depth': [3, 5],
            'subsample': [0.8, 1.0]
        }
    }
}

# 时间序列交叉验证配置
tscv = TimeSeriesSplit(n_splits=5)