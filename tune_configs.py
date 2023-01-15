from os import cpu_count
from ray import tune

from sklearn.linear_model import Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor


RIDGE_CONFIG = {
    "search_space": {"alpha": tune.loguniform(1e-4, 1e4)},
    "resources": {"cpu": 1},
}

LASSO_CONFIG = {
    "search_space": {"alpha": tune.loguniform(1e-4, 1e4)},
    "resources": {"cpu": 1},
}

RF_CONFIG = {
    "search_space": {
        "criterion": "squared_error",
        "random_state": 0,
        "n_jobs": cpu_count(),
        "n_estimators": tune.lograndint(1, 100),
        "max_depth": tune.randint(1, 10),
        "min_samples_split": tune.uniform(0, 0.5),
        "min_samples_leaf": tune.uniform(0, 0.5),
        "max_features": tune.uniform(0.5, 1.0),
    },
    "resources": {"cpu": cpu_count()},
}

XGB_CONFIG = {
    "search_space": {
        "objective": "reg:squarederror",
        "eval_metric": ["rmse"],
        "seed": 0,
        "n_jobs": cpu_count(),
        "n_estimators": tune.lograndint(1, 100),
        "max_depth": tune.randint(1, 10),
        "min_child_weight": tune.lograndint(1, 100),
        "subsample": tune.uniform(0.5, 1.0),
        "colsample_bytree": tune.uniform(0.5, 1.0),
        "lambda": tune.loguniform(1e-4, 1e4),
        "alpha": tune.loguniform(1e-4, 1e4),
        "eta": tune.loguniform(1e-3, 3e-1),
    },
    "resources": {"cpu": cpu_count()},
}

LGBM_CONFIG = {
    "search_space": {
        "objective": "regression",
        "seed": 0,
        "num_threads": cpu_count(),
        "n_estimators": tune.lograndint(1, 100),
        "max_depth": tune.randint(1, 10),
        "num_leaves": tune.lograndint(2, 100),
        "feature_fraction": tune.uniform(0.5, 1),
        "lambda_l1": tune.loguniform(1e-4, 1e4),
        "lambda_l2": tune.loguniform(1e-4, 1e4),
        "min_child_weight": tune.choice([1, 2, 3]),
        "subsample": tune.uniform(0.5, 1),
        "learning_rate": tune.loguniform(1e-3, 3e-1),
    },
    "resources": {"cpu": cpu_count()},
}


MODELS_CONFIG = {
    "Ridge": (Ridge, RIDGE_CONFIG),
    "Lasso": (Lasso, LASSO_CONFIG),
    "RF": (RandomForestRegressor, RF_CONFIG),
    "XGB": (XGBRegressor, XGB_CONFIG),
    "LGBM": (LGBMRegressor, LGBM_CONFIG),
}