import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
import joblib
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

os.makedirs("data/processed", exist_ok=True)
os.makedirs("models", exist_ok=True)

with open("params.yaml", "r") as f:
    params = yaml.safe_load(f)

raw_path = params["data"]["raw_path"]
test_size = params["data"]["test_size"]
random_state = params["data"]["split_random_state"]

df = pd.read_csv(raw_path)

df.drop("customerID", axis=1, inplace=True)
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})
df.dropna(inplace=True)
df.reset_index(drop=True, inplace=True)

binary_cols = [
    col for col in df.columns
    if df[col].dtype == "O" and df[col].nunique() == 2
]

le = LabelEncoder()
for col in binary_cols:
    df[col] = le.fit_transform(df[col])

ohe_cols = [col for col in df.columns if df[col].dtype == "O"]
df = pd.get_dummies(df, columns=ohe_cols, drop_first=True)

bool_cols = df.select_dtypes(include="bool").columns
df[bool_cols] = df[bool_cols].astype(int)

X = df.drop("Churn", axis=1)
y = df["Churn"]

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=test_size,
    random_state=random_state,
    stratify=y,
)

num_cols = ["tenure", "MonthlyCharges", "TotalCharges"]

scaler = StandardScaler()
X_train[num_cols] = scaler.fit_transform(X_train[num_cols])
X_test[num_cols] = scaler.transform(X_test[num_cols])

train_df = X_train.copy()
train_df["Churn"] = y_train.values

test_df = X_test.copy()
test_df["Churn"] = y_test.values

train_df.to_csv("data/processed/train.csv", index=False)
test_df.to_csv("data/processed/test.csv", index=False)

joblib.dump(scaler, "models/scaler.pkl")

print("Saved:")
print("  data/processed/train.csv")
print("  data/processed/test.csv")
print("  models/scaler.pkl")