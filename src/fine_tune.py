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
data_version = "fd9a77647fe6737064c2ba609042188a"
base_namespace = "model_finetuning"

################################
# part 1: XGB model finetuning #
################################

# (neptune) fetch project
project = neptune.get_project(name="common/project-tabular-data")

# (neptune) find best run for given data version
best_run_df = project.fetch_runs_table(tag="best").to_pandas()
best_run_df = best_run_df[best_run_df["data/train/version"] == data_version]
best_run_id = best_run_df["sys/id"].values[0]

# (neptune) resume this run
run = neptune.init(
    project="common/project-tabular-data",
    run=best_run_id,
    monitoring_namespace=f"{base_namespace}/monitoring",
)

# (neptune-xgboost integration) create neptune_callback to track XGBoost finetuning
neptune_callback = NeptuneCallback(
    run=run,
    base_namespace=base_namespace,
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
    "eta": 0.27,
    "gamma": 0,
    "max_depth": 8,
    "colsample_bytree": 0.7,
    "subsample": 0.8,
    "objective": "reg:squarederror",
    "eval_metric": ["mae", "rmse"],
}
evals = [(dtrain, "train"), (dtest, "valid")]
num_round = 200

# (neptune) download model from the run and start finetuning
run["model_training/pickled_model"].download("xgb.model")
with open("xgb.model", "rb") as file:
    bst = pickle.load(file)

bst_fine = xgb.train(
    params=model_params,
    dtrain=dtrain,
    num_boost_round=num_round,
    evals=evals,
    callbacks=[neptune_callback],
    xgb_model=bst,
)

test_preds = bst_fine.predict(dtest)

# (neptune) log test scores
run[f"{base_namespace}/test_score/rmse"] = np.sqrt(mean_squared_error(y_true=y_test, y_pred=test_preds))
run[f"{base_namespace}/test_score/mae"] = mean_absolute_error(y_true=y_test, y_pred=test_preds)

run["sys/tags"].add("finetuned")

run.sync(wait=True)

#########################################################################
# part 2: check if it's new best finetuned model for given data version #
#########################################################################

# (neptune) fetch necessary metadata from the current run
run_id = run["sys/id"].fetch()
test_rmse = run[f"{base_namespace}/test_score/rmse"].fetch()

# (neptune) find best-finetuned run for given data version
best_run_df = project.fetch_runs_table(tag="best-finetuned").to_pandas()
best_run_df = best_run_df[best_run_df["data/train/version"] == data_version]
best_rmse = best_run_df[f"{base_namespace}/test_score/rmse"].to_numpy()

# check if new model is new best
if test_rmse < best_rmse:
    run["sys/tags"].add("best-finetuned")

    best_run_id = best_run_df["sys/id"].values[0]
    best_run = neptune.init(
        project="common/project-tabular-data",
        run=best_run_id
    )
    best_run["sys/tags"].remove("best-finetuned")
