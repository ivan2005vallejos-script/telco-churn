"""Entrena una regresión logística de churn sobre la tabla customers de DuckDB."""

import json
from pathlib import Path

import duckdb
import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
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

DB_PATH = "data/warehouse/telco.duckdb"
MODEL_PATH = "models/churn_model.joblib"
METRICS_PATH = "reports/metrics.json"

TARGET = "Churn"
DROP_COLS = ["customerID"]
NUMERIC_COLS = ["tenure", "MonthlyCharges", "TotalCharges"]
PASSTHROUGH_COLS = ["SeniorCitizen"]
CATEGORICAL_COLS = [
    "gender",
    "Partner",
    "Dependents",
    "PhoneService",
    "MultipleLines",
    "InternetService",
    "OnlineSecurity",
    "OnlineBackup",
    "DeviceProtection",
    "TechSupport",
    "StreamingTV",
    "StreamingMovies",
    "Contract",
    "PaperlessBilling",
    "PaymentMethod",
]


def load_customers() -> pd.DataFrame:
    con = duckdb.connect(DB_PATH, read_only=True)
    df = con.execute("SELECT * FROM customers").fetchdf()
    con.close()
    return df


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("numeric", StandardScaler(), NUMERIC_COLS),
            ("passthrough", "passthrough", PASSTHROUGH_COLS),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_COLS,
            ),
        ]
    )
    return Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            (
                "classifier",
                LogisticRegression(class_weight="balanced", max_iter=1000),
            ),
        ]
    )


def split_data(df: pd.DataFrame):
    y = df[TARGET]
    X = df.drop(columns=DROP_COLS + [TARGET])
    return train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)


def main() -> None:
    df = load_customers()

    X_train, X_test, y_train, y_test = split_data(df)

    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    report = classification_report(y_test, y_pred, output_dict=True)
    cm = confusion_matrix(y_test, y_pred)

    metrics = {
        "precision_churn": precision_score(y_test, y_pred),
        "recall_churn": recall_score(y_test, y_pred),
        "f1_churn": f1_score(y_test, y_pred),
        "roc_auc": roc_auc_score(y_test, y_proba),
        "confusion_matrix": cm.tolist(),
        "classification_report": report,
    }

    print(classification_report(y_test, y_pred))
    print(f"Precision (churn): {metrics['precision_churn']:.4f}")
    print(f"Recall (churn):    {metrics['recall_churn']:.4f}")
    print(f"F1 (churn):        {metrics['f1_churn']:.4f}")
    print(f"ROC-AUC:           {metrics['roc_auc']:.4f}")
    print("Matriz de confusión (filas=real, columnas=pred):")
    print(cm)

    Path(MODEL_PATH).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(pipeline, MODEL_PATH)

    Path(METRICS_PATH).parent.mkdir(parents=True, exist_ok=True)
    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)


if __name__ == "__main__":
    main()
