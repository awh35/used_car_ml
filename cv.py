from functools import partial
import logging
import json

import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
import ray
from ray import tune
from ray.air import session
from ray.air.config import RunConfig

logger = logging.getLogger(__name__)

RANDOM_SEARCH_STEPS = 50
RAY_SPILL_PATH = "/mnt/disks/disk1/ray_spill"  # where to redirect memory spillage

def init_ray(ray_spill_path=RAY_SPILL_PATH):
    ray.init(
        ignore_reinit_error=True,
        _system_config={
            "object_spilling_config": json.dumps(
                {
                    "type": "filesystem",
                    "params": {"directory_path": ray_spill_path},
                }
            )
        },
    )


def construct_ts_folds(dts, n_splits, gap, max_train_size=None):
    """Returns a tuple of train-test folds which respect time order.

    Args:
        - dts: pandas series of datetimes to split
        - gap: gap left between train and test sets
    """
    split_idxs = TimeSeriesSplit(
        n_splits=n_splits, max_train_size=max_train_size, gap=gap
    ).split(dts)
    return tuple(((dts[train], dts[test]) for (train, test) in split_idxs))


def fit_and_predict_fold(
    train_dates, test_dates, config, model, df, feature_names, target_name, date_name
):
    """Fit model on the train set, then predict on the test set.
    Returns test set predictions.

    Args:
        - config: dict to pass as kwargs to model
        - model: model class to be fit, e.g. Ridge
        - feature_names: the column names of df to be used as features
        - target_name: the column name in df corresponding to the targets
        - date_name: the column name in df corresponding to the date
    """
    train_idx = df.loc[df[date_name].isin(train_dates)].index
    test_idx = df.loc[df[date_name].isin(test_dates)].index
    X_train, X_test = df.loc[train_idx, feature_names], df.loc[test_idx, feature_names]
    y_train = df.loc[train_idx, target_name]
    model_ = model(**config)
    model_.fit(X_train, y_train)
    return pd.Series(model_.predict(X_test), index=test_idx)


def fit_and_predict_all_folds(
    config,
    model,
    df,
    folds,
    feature_names,
    target_name,
    date_name,
    score_func,
):
    fit_and_predict_fold_ = partial(
        fit_and_predict_fold,
        config=config,
        model=model,
        df=df,
        feature_names=feature_names,
        target_name=target_name,
        date_name=date_name,
    )
    preds = [fit_and_predict_fold_(*fold) for fold in folds]
    test_scores = pd.Series(
        [score_func(df[target_name].reindex(y_pred.index), y_pred) for y_pred in preds]
    )
    return test_scores


def report_cv_score(
    config,
    model,
    df,
    folds,
    feature_names,
    target_name,
    date_name,
    score_func,
):
    oos_scores = fit_and_predict_all_folds(
        config,
        model,
        df,
        folds,
        feature_names,
        target_name,
        date_name,
        score_func,
    )
    session.report({"score": oos_scores.mean()})


def tune_hyperparameters(
    model,
    df,
    folds,
    config,
    target_name,
    feature_names,
    date_name,
    score_func,
    search_steps,
):
    report_cv_score_ = tune.with_parameters(
        report_cv_score,
        model=model,
        df=df,
        folds=folds,
        feature_names=feature_names,
        target_name=target_name,
        date_name=date_name,
        score_func=score_func,
    )
    report_cv_score_ = tune.with_resources(report_cv_score_, config["resources"])
    logger.info("Starting hyperparameter tuning.")
    tuner = tune.Tuner(
        report_cv_score_,
        tune_config=tune.TuneConfig(
            metric="score",
            mode="max",
            num_samples=search_steps,
        ),
        param_space=config["search_space"],
        run_config=RunConfig(verbose=1),
    )
    logger.info("Hyperparameter tuning completed.")
    return tuner


def tune_all_models(
    data,
    folds,
    models_config,
    target_name,
    feature_names,
    date_name,
    score_func,
    search_steps=RANDOM_SEARCH_STEPS,
):
    init_ray()
    tune_hyperparameters_ = partial(
        tune_hyperparameters,
        target_name=target_name,
        feature_names=feature_names,
        date_name=date_name,
        score_func=score_func,
        search_steps=search_steps,
    )
    tuner_dict = {
        k: tune_hyperparameters_(model, data, folds, search_space)
        for k, (model, search_space) in models_config.items()
    }
    results_dict = {k: tuner.fit() for k, tuner in tuner_dict.items()}
    return {k: res.get_best_result() for k, res in results_dict.items()}
