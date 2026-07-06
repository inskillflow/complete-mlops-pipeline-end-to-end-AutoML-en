import argparse
import logging
import os
import warnings

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, Lasso, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

logging.basicConfig(level=logging.WARN)
logger = logging.getLogger(__name__)

parser = argparse.ArgumentParser()
parser.add_argument("--alpha", type=float, required=False, default=0.7)
parser.add_argument("--l1_ratio", type=float, required=False, default=0.7)
args = parser.parse_args()


def eval_metrics(actual, pred):
    rmse = np.sqrt(mean_squared_error(actual, pred))
    mae = mean_absolute_error(actual, pred)
    r2 = r2_score(actual, pred)
    return rmse, mae, r2


COMMON_TAGS = {
    "engineering":       "ML platform",
    "release.candidate": "RC1",
    "release.version":   "2.0",
}


def make_elasticnet(alpha, l1_ratio):
    return (ElasticNet(alpha=alpha, l1_ratio=l1_ratio, random_state=42),
            {"alpha": alpha, "l1_ratio": l1_ratio})


def make_ridge(alpha, l1_ratio):
    return (Ridge(alpha=alpha, random_state=42),
            {"alpha": alpha})


def make_lasso(alpha, l1_ratio):
    return (Lasso(alpha=alpha, random_state=42),
            {"alpha": alpha})


def train_one_run(run_name, factory, alpha, l1_ratio,
                  train_x, train_y, test_x, test_y):
    """Train ONE model (whatever the factory returns) and log it to MLflow."""
    mlflow.start_run(run_name=run_name)
    mlflow.set_tags(COMMON_TAGS)

    current = mlflow.active_run()
    print(f"  >>> {current.info.run_name}  (id={current.info.run_id})")

    estimator, params_to_log = factory(alpha, l1_ratio)
    estimator.fit(train_x, train_y)
    preds = estimator.predict(test_x)
    rmse, mae, r2 = eval_metrics(test_y, preds)

    print(f"      {type(estimator).__name__}({params_to_log})  "
          f"RMSE={rmse:.4f}  MAE={mae:.4f}  R2={r2:.4f}")

    mlflow.log_params(params_to_log)
    mlflow.log_metrics({"rmse": rmse, "r2": r2, "mae": mae})
    mlflow.sklearn.log_model(estimator, "my_new_model_1")
    mlflow.log_artifacts("data/")

    mlflow.end_run()


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    np.random.seed(40)

    mlflow.set_tracking_uri(
        os.getenv("MLFLOW_TRACKING_URI", "http://mlflow:5000")
    )
    print("The set tracking URI is", mlflow.get_tracking_uri())

    data = pd.read_csv("data/red-wine-quality.csv")
    train, test = train_test_split(data)

    os.makedirs("data", exist_ok=True)
    train.to_csv("data/train.csv", index=False)
    test.to_csv("data/test.csv", index=False)

    train_x = train.drop(["quality"], axis=1)
    test_x = test.drop(["quality"], axis=1)
    train_y = train[["quality"]]
    test_y = test[["quality"]]

    EXPERIMENTS = [
        ("exp_multi_EL",    make_elasticnet),
        ("exp_multi_Ridge", make_ridge),
        ("exp_multi_Lasso", make_lasso),
    ]
    ALPHAS = [args.alpha, 0.9, 0.4]

    for exp_name, factory in EXPERIMENTS:
        print(f"\n========== Experiment: {exp_name} ==========")
        exp = mlflow.set_experiment(experiment_name=exp_name)
        print(f"  Name: {exp.name}  Experiment_id: {exp.experiment_id}")

        for i, alpha in enumerate(ALPHAS, start=1):
            train_one_run(
                run_name=f"run{i}.1",
                factory=factory,
                alpha=alpha,
                l1_ratio=args.l1_ratio,
                train_x=train_x, train_y=train_y,
                test_x=test_x, test_y=test_y,
            )

    run = mlflow.last_active_run()
    print(f"\nRecent active run id   : {run.info.run_id}")
    print(f"Recent active run name : {run.info.run_name}")
