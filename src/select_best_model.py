import os
import json
import shutil
import pandas as pd
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)



os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)

with open("results/random_forest_metrics.json", "r") as f:
    rf_metrics = json.load(f)

with open("results/catboost_metrics.json", "r") as f:
    cb_metrics = json.load(f)

comparison = pd.DataFrame({
    "RandomForest": rf_metrics,
    "CatBoost": cb_metrics,
}).T

comparison.to_csv("results/model_comparison.csv")

if cb_metrics["f1"] > rf_metrics["f1"]:
    best_model_name = "CatBoost"
    best_model_path = "models/catboost.pkl"
else:
    best_model_name = "RandomForest"
    best_model_path = "models/random_forest.pkl"

shutil.copy(best_model_path, "models/best_model.pkl")

with open("results/best_model.json", "w") as f:
    json.dump(
        {
            "best_model": best_model_name,
            "selection_metric": "f1",
            "best_f1": max(cb_metrics["f1"], rf_metrics["f1"]),
        },
        f,
        indent=4,
    )

print(f"Best model: {best_model_name}")