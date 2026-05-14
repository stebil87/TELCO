import os
import pandas as pd
from deepchecks.tabular import Dataset
from deepchecks.tabular.suites import data_integrity, train_test_validation

os.makedirs("deepchecks_reports", exist_ok=True)

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]
feature_cols = [c for c in train_df.columns if c != "Churn"]
cat_features = [c for c in feature_cols if c not in num_cols]

ds_train = Dataset(train_df, label="Churn", cat_features=cat_features)
ds_test = Dataset(test_df, label="Churn", cat_features=cat_features)

integrity_result = data_integrity().run(ds_train)
integrity_result.save_as_html("deepchecks_reports/data_integrity.html")

train_test_result = train_test_validation().run(ds_train, ds_test)
train_test_result.save_as_html("deepchecks_reports/train_test_validation.html")

print("Saved Deepchecks data reports.")