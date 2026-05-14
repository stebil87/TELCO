import os
import json
import yaml
import joblib
import mlflow
import mlflow.catboost
import pandas as pd

from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

X_train = train_df.drop("Churn", axis=1)
y_train = train_df["Churn"]

X_test = test_df.drop("Churn", axis=1)
y_test = test_df["Churn"]

mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("telco_churn")

cb_params = params["catboost"]

with mlflow.start_run(run_name="CatBoost_tuned"):

    model = CatBoostClassifier(
        iterations=cb_params["iterations"],
        learning_rate=cb_params["learning_rate"],
        depth=cb_params["depth"],
        l2_leaf_reg=cb_params["l2_leaf_reg"],
        auto_class_weights="Balanced",
        eval_metric="F1",
        random_seed=cb_params["random_seed"],
        verbose=100,
    )

    model.fit(
        X_train,
        y_train,
        eval_set=(X_test, y_test),
        early_stopping_rounds=cb_params["early_stopping_rounds"],
    )

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
    }

    joblib.dump(model, "models/catboost.pkl")

    with open("results/catboost_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    feat_imp = pd.DataFrame({
        "feature": X_train.columns,
        "importance": model.get_feature_importance(),
    }).sort_values("importance", ascending=False)

    feat_imp.to_csv("results/feature_importance.csv", index=False)

    mlflow.log_params(cb_params)
    mlflow.log_metrics(metrics)
    mlflow.catboost.log_model(model, "catboost_model")

print("Saved CatBoost model, metrics and feature importance.")