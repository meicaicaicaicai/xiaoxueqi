# 模型定义、超参搜索、训练

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import GridSearchCV, TimeSeriesSplit
from configuration import GB_PARAM_GRID, XGB_PARAM_GRID, RANDOM_STATE

def build_models():
    tscv = TimeSeriesSplit(n_splits=5)

    models = {
        'GradientBoosting':
            GridSearchCV(
                GradientBoostingRegressor(random_state=RANDOM_STATE),
                param_grid=GB_PARAM_GRID,
                cv=tscv, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            ),
        'XGBoost':
            GridSearchCV(
                XGBRegressor(objective='reg:squarederror', random_state=RANDOM_STATE, n_jobs=-1),
                param_grid=XGB_PARAM_GRID,
                cv=tscv, scoring='neg_mean_squared_error',
                n_jobs=-1, verbose=1
            )
    }
    return models