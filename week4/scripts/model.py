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