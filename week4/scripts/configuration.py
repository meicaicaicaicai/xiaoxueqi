# 有哪些模型、地基怎么搭、上层怎么选。
# main.py 和 model.py 只负责“点菜”和“做菜”，
# 随时可以加菜、换调料、换做法，而不动厨房结构。

from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from sklearn.model_selection import TimeSeriesSplit
from lightgbm import LGBMRegressor

# conf={}这是个大字典，键是模型名字符串，值是二级字典，含2个固定键
conf = {
    # "GradientBoosting"这个是键
    "GradientBoosting": {
        "model_name": GradientBoostingRegressor,
        "model_params": {
            "random_state": 42
        },
        # "param_grid": {
        #     'n_estimators': [100, 200],
        #     'learning_rate': [0.05, 0.1],
        #     'max_depth': [3, 5],
        #     'min_samples_split': [2, 5]
        # }
    },
    "XGBoost": {
        "model_name": XGBRegressor,
        "model_params": {
            "objective": 'reg:squarederror',
            "random_state": 42,
            "n_jobs": -1
        },
        # "param_grid": {
        #     'n_estimators': [100, 200],
        #     'learning_rate': [0.05, 0.1],
        #     'max_depth': [3, 5],
        #     'subsample': [0.8, 1.0]
        # }
    },
    "LightGBM": {
        # 图纸，类本身
        # model_cls = LGBMRegressor
        # 房子，实例
        # model_obj = LGBMRegressor(random_state=42, n_estimators=100)
        "model_name": LGBMRegressor,

        # 地基，可以随意加减
        "model_params": {
            "random_state": 42,
            "n_jobs": -1,
            "verbose": -1
        },

        # # 网格搜索候选值就是把所有组合搞一遍，找出最好的那个
        # "param_grid": {
        #     # 树的数量
        #     'n_estimators': [100, 200],
        #     # 每棵树的学习步长
        #     'learning_rate': [0.05, 0.1],
        #     # 单棵树最大深度
        #     'max_depth': [3, 5],
        #     # 每棵树用的样本比例
        #     'subsample': [0.8, 1.0]
        # }
    }
}

# 时间序列交叉验证配置
tscv = TimeSeriesSplit(n_splits=5)
# TimeSeriesSplit：因为数据按日期排序，未来不能泄漏到过去，主要是时间轴滚动验证防泄漏

# 其他切分方法：KFold（普通交叉验证）
# StratifiedKFold（分类任务）
# TimeSeriesSplit（时间序列专用，已在用）

