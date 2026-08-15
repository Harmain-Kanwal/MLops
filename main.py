"""
FastAPI service for the Customer Churn Prediction model.
Run: uvicorn main:app --reload
"""
import csv
import json
import os
import time
from datetime import datetime, timezone
from typing import Literal

import joblib
import pandas as pd
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

MODEL_PATH = "models/model.pkl"
STATS_PATH = "models/train_stats.json"
LOG_PATH = "predictions_log.csv"
DRIFT_THRESHOLD_PCT = 20

app = FastAPI(title="Customer Churn Prediction API", version="1.1.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

model = joblib.load(MODEL_PATH)
with open(STATS_PATH) as f:
    train_stats = json.load(f)


class CustomerFeatures(BaseModel):
    gender: Literal["Male", "Female"]
    SeniorCitizen: Literal[0, 1]
    Partner: Literal["Yes", "No"]
    Dependents: Literal["Yes", "No"]
    tenure: int = Field(..., ge=0, le=100)
    PhoneService: Literal["Yes", "No"]
    MultipleLines: Literal["Yes", "No", "No phone service"]
    InternetService: Literal["DSL", "Fiber optic", "No"]
    OnlineSecurity: Literal["Yes", "No", "No internet service"]
    OnlineBackup: Literal["Yes", "No", "No internet service"]
    DeviceProtection: Literal["Yes", "No", "No internet service"]
    TechSupport: Literal["Yes", "No", "No internet service"]
    StreamingTV: Literal["Yes", "No", "No internet service"]
    StreamingMovies: Literal["Yes", "No", "No internet service"]
    Contract: Literal["Month-to-month", "One year", "Two year"]
    PaperlessBilling: Literal["Yes", "No"]
    PaymentMethod: Literal[
        "Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"
    ]
    MonthlyCharges: float = Field(..., ge=0)
    TotalCharges: float = Field(..., ge=0)


def explain_prediction(customer: dict) -> list:
    """Rank the features that pushed this specific prediction toward churn or staying,
    using real churn rates from the training data (not a black box)."""
    baseline = train_stats.get("baseline_churn_rate", 0.0)
    cat_rates = train_stats.get("category_churn_rates", {})
    col_importance = train_stats.get("column_importance", {})
    numeric_corr = train_stats.get("numeric_correlations", {})
    numeric_means = {k: float(v) for k, v in train_stats.get("numeric_means", {}).items()}

    factors = []
    for col, rates in cat_rates.items():
        value = customer.get(col)
        if value in rates:
            rate = float(rates[value])
            delta = rate - baseline
            weight = float(col_importance.get(col, 0.0))
            factors.append({
                "feature": col,
                "pushes_toward": "churn" if delta > 0 else "staying",
                "explanation": f"{col} = '{value}': {rate*100:.0f}% of similar customers churn vs {baseline*100:.0f}% average",
                "impact": abs(delta) * weight,
            })

    for col, corr in numeric_corr.items():
        value = customer.get(col)
        mean = numeric_means.get(col)
        if value is None or mean is None:
            continue
        diff = value - mean
        pushes_churn = (diff > 0 and corr > 0) or (diff < 0 and corr < 0)
        weight = float(col_importance.get(col, 0.0))
        cmp_word = "higher" if diff > 0 else "lower"
        factors.append({
            "feature": col,
            "pushes_toward": "churn" if pushes_churn else "staying",
            "explanation": f"{col} = {value:g}: {cmp_word} than the average of {mean:.0f}",
            "impact": abs(diff) * abs(corr) * weight,
        })

    factors.sort(key=lambda f: f["impact"], reverse=True)
    for f in factors:
        del f["impact"]
    return factors[:4]


def log_prediction(payload: dict, prediction: int, probability: float, latency_ms: float) -> None:
    file_exists = os.path.isfile(LOG_PATH)
    with open(LOG_PATH, "a", newline="") as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(["timestamp"] + list(payload.keys()) + ["prediction", "probability", "latency_ms"])
        writer.writerow(
            [datetime.now(timezone.utc).isoformat()] + list(payload.values()) + [prediction, round(probability, 4), round(latency_ms, 2)]
        )


@app.get("/")
def root():
    return {"message": "Customer Churn Prediction API is running. See /docs for the interactive API explorer."}


@app.get("/health")
def health():
    return {"status": "ok", "model_loaded": model is not None}


@app.post("/predict")
def predict(customer: CustomerFeatures):
    start = time.time()
    try:
        payload = customer.model_dump()
        input_df = pd.DataFrame([payload])
        pred = int(model.predict(input_df)[0])
        prob = float(model.predict_proba(input_df)[0][1])
        top_factors = explain_prediction(payload)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

    latency_ms = (time.time() - start) * 1000
    log_prediction(payload, pred, prob, latency_ms)

    return {
        "churn_prediction": "Yes" if pred == 1 else "No",
        "churn_probability": round(prob, 4),
        "latency_ms": round(latency_ms, 2),
        "top_factors": top_factors,
    }


@app.get("/metrics")
def metrics():
    if not os.path.isfile(LOG_PATH):
        return {"total_predictions": 0, "message": "No predictions logged yet.", "training_metrics": train_stats.get("metrics", {})}

    df = pd.read_csv(LOG_PATH)
    total = len(df)
    churn_rate = float((df["prediction"] == 1).mean())
    avg_latency = float(df["latency_ms"].mean())

    drift = {}
    for col in train_stats["numeric_cols"]:
        if col in df.columns:
            live_mean = float(df[col].mean())
            train_mean = float(train_stats["numeric_means"].get(col, live_mean))
            pct_diff = abs(live_mean - train_mean) / (abs(train_mean) + 1e-9) * 100
            drift[col] = {
                "train_mean": round(train_mean, 2),
                "live_mean": round(live_mean, 2),
                "pct_diff": round(pct_diff, 2),
                "drift_alert": pct_diff > DRIFT_THRESHOLD_PCT,
            }

    return {
        "total_predictions": total,
        "churn_rate": round(churn_rate, 4),
        "avg_latency_ms": round(avg_latency, 2),
        "drift": drift,
        "training_metrics": train_stats.get("metrics", {}),
    }