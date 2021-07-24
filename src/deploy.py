import neptune.new as neptune

data_version = "5b49383080a7edfe4ef72dc359112d3c"
base_namespace = "production"

# (neptune) fetch project
project = neptune.get_project(name="common/project-tabular-data")

# (neptune) find best run for given data version
best_run_df = project.fetch_runs_table(tag="best-finetuned").to_pandas()
best_run_df = best_run_df[best_run_df["data/train/version"] == data_version]
best_run_id = best_run_df["sys/id"].values[0]

# (neptune) resume this run
run = neptune.init(
    project="common/project-tabular-data",
    run=best_run_id,
    capture_hardware_metrics=False,
)

# (neptune) download model from the run
run["model_training/pickled_model"].download("xgb.model")

# here goes deploying logic

# (neptune) log model version that is now in prod
in_prod_run_df = project.fetch_runs_table(tag="in-prod").to_pandas()
in_prod_run_df = in_prod_run_df[in_prod_run_df["data/train/version"] == data_version]
in_prod_run_id = in_prod_run_df["sys/id"].values[0]

# (neptune) resume in-prod run
run_in_prod = neptune.init(
    project="common/project-tabular-data",
    run=in_prod_run_id,
    capture_hardware_metrics=False,
)

# increment model version
model_version = run_in_prod[f"{base_namespace}/model_version"].fetch()
run[f"{base_namespace}/model_version"] = "xgb-{}".format(int(model_version.split("-")[-1]) + 1)

# (neptune) move "in-prod" tag to the new run
run_in_prod["sys/tags"].remove("in-prod")
run["sys/tags"].add("in-prod")
