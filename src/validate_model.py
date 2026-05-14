import os
import joblib
import mlflow
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import model_evaluation

import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

os.makedirs("deepchecks_reports", exist_ok=True)

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

model = joblib.load("models/best_model.pkl")

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
feature_cols = [c for c in train_df.columns if c != "Churn"]
cat_features = [c for c in feature_cols if c not in num_cols]

ds_train = Dataset(train_df, label="Churn", cat_features=cat_features)
ds_test = Dataset(test_df, label="Churn", cat_features=cat_features)

model_eval_result = model_evaluation().run(ds_train, ds_test, model)
model_eval_result.save_as_html("deepchecks_reports/model_evaluation.html")

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("telco_churn")

with mlflow.start_run(run_name="deepchecks_best_model"):
    mlflow.log_artifact("deepchecks_reports/model_evaluation.html")

print("Saved Deepchecks model evaluation report.")