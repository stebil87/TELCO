import os
import json
import yaml
import joblib
import mlflow
import mlflow.sklearn
import pandas as pd

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedKFold, cross_validate
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

rf_params = {
    "n_estimators": [100, 200, 300],
    "max_depth": [8, 12, None],
    "min_samples_split": [5, 10],
    "class_weight": ["balanced"],
}

cv = StratifiedKFold(
    n_splits=params["random_forest"]["cv_splits"],
    shuffle=True,
    random_state=params["data"]["split_random_state"],
)

with mlflow.start_run(run_name="RandomForest_tuned"):

    grid = GridSearchCV(
        RandomForestClassifier(random_state=params["random_forest"]["random_state"]),
        param_grid=rf_params,
        cv=cv,
        scoring="f1",
        n_jobs=-1,
    )

    grid.fit(X_train, y_train)
    model = grid.best_estimator_

    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "best_cv_f1": grid.best_score_,
    }

    cv_results = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "f1", "recall", "roc_auc"],
    )

    for metric_name in ["accuracy", "f1", "recall", "roc_auc"]:
        metrics[f"cv_{metric_name}_mean"] = cv_results[f"test_{metric_name}"].mean()
        metrics[f"cv_{metric_name}_std"] = cv_results[f"test_{metric_name}"].std()

    joblib.dump(model, "models/random_forest.pkl")

    with open("results/random_forest_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    mlflow.log_params(grid.best_params_)
    mlflow.log_metrics(metrics)
    mlflow.sklearn.log_model(model, "random_forest_model")

print("Saved Random Forest model and metrics.")