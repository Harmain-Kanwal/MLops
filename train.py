"""
Train a Customer Churn Prediction model and track the experiment with MLflow.
Run: python train.py
"""
import json
import os

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# ---------- 1. Load & clean data ----------
DATA_PATH = "data/telco_churn.csv"
df = pd.read_csv(DATA_PATH)

df = df.drop(columns=["customerID"])
df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

TARGET = "Churn"
df[TARGET] = df[TARGET].map({"Yes": 1, "No": 0})

X = df.drop(columns=[TARGET])
y = df[TARGET]

categorical_cols = X.select_dtypes(include="object").columns.tolist()
numeric_cols = X.select_dtypes(exclude="object").columns.tolist()

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# ---------- 2. Build the pipeline ----------
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), numeric_cols),
        ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols),
    ]
)

params = {
    "n_estimators": 200,
    "max_depth": 8,
    "min_samples_leaf": 5,
    "random_state": 42,
    "class_weight": "balanced",
}

model = Pipeline(
    steps=[
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(**params)),
    ]
)

# ---------- 3. Train + track with MLflow ----------
mlflow.set_tracking_uri("sqlite:///mlflow.db")
mlflow.set_experiment("customer_churn_prediction")

with mlflow.start_run(run_name="random_forest_baseline") as run:
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    probs = model.predict_proba(X_test)[:, 1]

    metrics = {
        "accuracy": accuracy_score(y_test, preds),
        "precision": precision_score(y_test, preds),
        "recall": recall_score(y_test, preds),
        "f1_score": f1_score(y_test, preds),
        "roc_auc": roc_auc_score(y_test, probs),
    }

    mlflow.log_params(params)
    mlflow.log_metrics(metrics)
    mlflow.log_param("n_train_rows", len(X_train))
    mlflow.log_param("n_test_rows", len(X_test))

    cm = confusion_matrix(y_test, preds)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=["No churn", "Churn"], yticklabels=["No churn", "Churn"])
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix - Churn Model")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png")
    mlflow.log_artifact("confusion_matrix.png")
    plt.close()

    ohe_cols = model.named_steps["preprocessor"].named_transformers_["cat"].get_feature_names_out(categorical_cols)
    all_feature_names = numeric_cols + list(ohe_cols)
    importances = model.named_steps["classifier"].feature_importances_
    top_idx = importances.argsort()[::-1][:15]
    plt.figure(figsize=(7, 6))
    plt.barh([all_feature_names[i] for i in top_idx][::-1], importances[top_idx][::-1])
    plt.title("Top 15 Feature Importances")
    plt.tight_layout()
    plt.savefig("feature_importance.png")
    mlflow.log_artifact("feature_importance.png")
    plt.close()

    mlflow.sklearn.log_model(model, "model", registered_model_name="churn_predictor")

    print(f"\nMLflow run ID: {run.info.run_id}")
    print("Metrics:", json.dumps(metrics, indent=2))
    print("\n" + classification_report(y_test, preds, target_names=["No churn", "Churn"]))

# ---------- 4. Explainability data (for the "why" behind each prediction) ----------
# Aggregate one-hot feature importances back to their original column names
feature_importance_map = dict(zip(all_feature_names, importances))
column_importance = {}
for col in numeric_cols:
    column_importance[col] = float(feature_importance_map.get(col, 0.0))
for col in categorical_cols:
    total = sum(v for k, v in feature_importance_map.items() if k.startswith(f"{col}_"))
    column_importance[col] = float(total)
ranked_features = sorted(column_importance, key=column_importance.get, reverse=True)

# Real churn rate for every category value, e.g. Contract="Month-to-month" -> 0.43
train_with_target = X_train.copy()
train_with_target["Churn"] = y_train.values
category_churn_rates = {
    col: {str(k): float(v) for k, v in train_with_target.groupby(col)["Churn"].mean().items()}
    for col in categorical_cols
}
baseline_churn_rate = float(y_train.mean())

# Correlation direction for numeric features: positive = higher value -> more churn
numeric_correlations = {
    col: float(train_with_target[col].corr(train_with_target["Churn"]))
    for col in numeric_cols
}

print(f"\nBaseline churn rate: {baseline_churn_rate*100:.1f}%")
print("Top 5 most predictive features:", ranked_features[:5])

# ---------- 5. Export model + stats for the FastAPI service ----------
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/model.pkl")

train_stats = {
    "numeric_cols": numeric_cols,
    "categorical_cols": categorical_cols,
    "numeric_means": X_train[numeric_cols].mean().to_dict(),
    "numeric_stds": X_train[numeric_cols].std().to_dict(),
    "categorical_values": {c: sorted(X_train[c].unique().tolist()) for c in categorical_cols},
    "metrics": metrics,
    "run_id": run.info.run_id,
    "ranked_features": ranked_features,
    "column_importance": column_importance,
    "category_churn_rates": category_churn_rates,
    "baseline_churn_rate": baseline_churn_rate,
    "numeric_correlations": numeric_correlations,
}
with open("models/train_stats.json", "w") as f:
    json.dump(train_stats, f, indent=2, default=str)

print("\nSaved model      -> models/model.pkl")
print("Saved stats       -> models/train_stats.json")
print("View experiments  -> run: mlflow ui --backend-store-uri sqlite:///mlflow.db")