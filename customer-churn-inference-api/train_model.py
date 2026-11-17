"""Train the demonstration pipeline from the supplied CSV."""
from pathlib import Path
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LogisticRegression

ROOT = Path(__file__).resolve().parent
DATA_PATH = ROOT / "data" / "customer-churn-training.csv"
MODEL_PATH = ROOT / "artifacts" / "churn_pipeline.joblib"

NUMERIC = ["tenure_months", "support_tickets", "monthly_spend_inr", "last_login_days"]
CATEGORICAL = ["plan_type"]
FEATURES = NUMERIC + CATEGORICAL
TARGET = "churned"

def main():
    data = pd.read_csv(DATA_PATH)
    required = set(FEATURES + [TARGET])
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")
    if data[TARGET].isna().any() or data[TARGET].nunique() != 2:
        raise ValueError("Target must contain two non-null classes.")
    if set(data[TARGET].unique()) - {0, 1}:
        raise ValueError("Target must be encoded as 0 or 1.")

    transformer = ColumnTransformer([
        ("numeric", Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]), NUMERIC),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]), CATEGORICAL),
    ])
    model = Pipeline([
        ("preprocess", transformer),
        ("classifier", LogisticRegression(
            class_weight="balanced", max_iter=1000, random_state=42
        )),
    ])
    model.fit(data[FEATURES], data[TARGET].astype(int))
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Saved demonstration model to {MODEL_PATH}")
    print(f"Training rows: {len(data)}")
    print("WARNING: this small sample does not validate production readiness.")

if __name__ == "__main__":
    main()
