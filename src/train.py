import hashlib
import pickle

import neptune.new as neptune
import numpy as np
import pandas as pd
import xgboost as xgb
from neptune.new.integrations.xgboost import NeptuneCallback
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.model_selection import train_test_split

rnd_state = 123

# (neptune) create run
run = neptune.init(
    project="common/project-tabular-data",
    name="training",
    tags=["xgb-integration", "experimenting"],
)

# (neptune-xgboost integration) create neptune callback to track XGBoost training
neptune_callback = NeptuneCallback(
    run=run,
    base_namespace="model_training",
    log_tree=[0, 1, 2]
)

# prepare data
data = pd.read_csv('../data/train.csv')
data.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = data.SalePrice
X = data.drop(['SalePrice'], axis=1).select_dtypes(exclude=['object'])
X_tr_va, X_test, y_tr_va, y_test = train_test_split(
    X.to_numpy(),
    y.to_numpy(),
    test_size=0.15,
    random_state=rnd_state
)
X_train, X_valid, y_train, y_valid = train_test_split(
    X_tr_va,
    y_tr_va,
    test_size=0.15,
    random_state=rnd_state
)
simple_imputer = SimpleImputer()
X_train = simple_imputer.fit_transform(X_train)
X_valid = simple_imputer.transform(X_valid)
X_test = simple_imputer.transform(X_test)

dtrain = xgb.DMatrix(X_train, label=y_train)
dval = xgb.DMatrix(X_valid, label=y_valid)
dtest = xgb.DMatrix(X_test, label=y_test)

# (neptune) log data version
run["data/train/version"] = hashlib.md5(X_train).hexdigest()
run["data/valid/version"] = hashlib.md5(X_valid).hexdigest()
run["data/test/version"] = hashlib.md5(X_test).hexdigest()

# (neptune) log datasets sizes
run["data/train/size"] = len(X_train)
run["data/valid/size"] = len(X_valid)
run["data/test/size"] = len(X_test)

# (neptune) log train sample
run["data/raw_sample"].upload(neptune.types.File.as_html(data.head(20)))

# define parameters
model_params = {
    "eta": 0.3,
    "gamma": 0.001,
    "max_depth": 4,
    "colsample_bytree": 0.7,
    "lambda": 1.5,
    "objective": "reg:squarederror",
    "eval_metric": ["mae", "rmse"],
}
evals = [(dtrain, "train"), (dval, "valid")]
num_round = 30

# (neptune) pass neptune_callback to the train function and run training
xgb.train(
    params=model_params,
    dtrain=dtrain,
    num_boost_round=num_round,
    evals=evals,
    callbacks=[
        neptune_callback,
        xgb.callback.LearningRateScheduler(lambda epoch: 0.99 ** (epoch+1)),
    ],
)
run.sync(wait=True)

run["model_training/pickled_model"].download("xgb.model")
with open("xgb.model", "rb") as file:
    bst = pickle.load(file)

test_preds = bst.predict(dtest)

# (neptune) log test scores
run["test_score/rmse"] = np.sqrt(mean_squared_error(y_true=y_test, y_pred=test_preds))
run["test_score/mae"] = mean_absolute_error(y_true=y_test, y_pred=test_preds)

# (neptune) determine if it is new best model so far and tag it "best"

# (neptune) Fetch project
project = neptune.get_project(name="common/project-tabular-data")

# (neptune) Find best run
runs_table_df = project.fetch_runs_table().to_pandas()
runs_table_df = runs_table_df.sort_values(by="model_training/valid/rmse", ascending=False)
run_id = runs_table_df["sys/id"].values[0]
