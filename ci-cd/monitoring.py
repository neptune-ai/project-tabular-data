import os
import random

import neptune.new as neptune

data_version = "5b49383080a7edfe4ef72dc359112d3c"
base_namespace = "production"

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
run = neptune.init(
    api_token=os.getenv('NEPTUNE_API_TOKEN'),
    project="common/project-tabular-data",
    run=in_prod_run_id,
    capture_hardware_metrics=False,
)

# (neptune) run monitoring logic
# ... and log metadata to the run
run[f"{base_namespace}/drift-score"].log(random.random() * 100)
