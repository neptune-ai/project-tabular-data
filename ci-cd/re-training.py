import os
import pickle

import neptune.new as neptune
import numpy as np
import pandas as pd
import xgboost as xgb
from neptune.new.integrations.xgboost import NeptuneCallback
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

rnd_state = 7937694
data_version = "5b49383080a7edfe4ef72dc359112d3c"
prod_namespace = "production"
retraining_namespace = "retraining"

# (neptune) fetch project
project = neptune.get_project(
    api_token=os.getenv('NEPTUNE_API_TOKEN'),
    name="common/project-tabular-data"
)

# (neptune) find run with "in-prod" tag for specific data version
in_prod_run_df = project.fetch_runs_table(tag="in-prod").to_pandas()
in_prod_run_df = in_prod_run_df[in_prod_run_df["data/train/version"] == data_version]
in_prod_run_id = in_prod_run_df["sys/id"].values[0]

# (neptune) resume this run
reference_run = neptune.init(
    api_token=os.getenv('NEPTUNE_API_TOKEN'),
    project="common/project-tabular-data",
    run=in_prod_run_id,
    capture_hardware_metrics=False,
    mode="read-only",
)

# (neptune) check if drift-score is below threshold
drift_score = reference_run[f"{prod_namespace}/drift-score"].fetch_last()

if drift_score < 0.95:
    print("Model performance Ok")
else:
    # (neptune) create run
    run = neptune.init(
        project="common/project-tabular-data",
        tags=["retraining"],
    )

    # (neptune) log reference run info
    run["reference_run_id"] = reference_run["sys/id"].fetch()

    # (neptune-xgboost integration) create neptune_callback to track XGBoost finetuning
    neptune_callback = NeptuneCallback(
        run=run,
        base_namespace=retraining_namespace,
        log_tree=[0, 1],
    )

    # prepare data
    data = pd.read_csv('../data/train.csv')
    data.dropna(axis=0, subset=['SalePrice'], inplace=True)
    y = data.SalePrice
    X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
    X_tr_va, X_test, y_tr_va, y_test = train_test_split(
        X.to_numpy(),
        y.to_numpy(),
        test_size=0.2,
        random_state=rnd_state
    )
    X_train, X_valid, y_train, y_valid = train_test_split(
        X_tr_va,
        y_tr_va,
        test_size=0.2,
        random_state=rnd_state
    )
    simple_imputer = SimpleImputer()
    X_train = simple_imputer.fit_transform(X_train)
    X_valid = simple_imputer.transform(X_valid)
    X_test = simple_imputer.transform(X_test)

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dval = xgb.DMatrix(X_valid, label=y_valid)
    dtest = xgb.DMatrix(X_test, label=y_test)

    # define parameters
    model_params = {
        "eta": 0.3,
        "gamma": 0.0001,
        "max_depth": 2,
        "colsample_bytree": 0.85,
        "subsample": 0.9,
        "objective": "reg:squarederror",
        "eval_metric": ["mae", "rmse"],
    }
    evals = [(dtrain, "train"), (dtest, "valid")]
    num_round = 200

    # (neptune) download model from the run and start finetuning
    reference_run["model_finetuning/pickled_model"].download("xgb.model")
    with open("xgb.model", "rb") as file:
        bst = pickle.load(file)

    bst_retrained = xgb.train(
        params=model_params,
        dtrain=dtrain,
        num_boost_round=num_round,
        evals=evals,
        callbacks=[neptune_callback],
        xgb_model=bst,
    )

    test_preds = bst_retrained.predict(dtest)

    # (neptune) log test scores
    run[f"{retraining_namespace}/test_score/rmse"] = np.sqrt(mean_squared_error(y_true=y_test, y_pred=test_preds))
    run[f"{retraining_namespace}/test_score/mae"] = mean_absolute_error(y_true=y_test, y_pred=test_preds)
