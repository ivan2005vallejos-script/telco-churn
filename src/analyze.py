"""Análisis del modelo de churn: coeficientes, predicciones en test y umbrales."""

from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import precision_score, recall_score

from train import MODEL_PATH, load_customers, split_data

COEFFICIENTS_PATH = "reports/coefficients.csv"
PREDICTIONS_PATH = "reports/test_predictions.csv"
THRESHOLD_PATH = "reports/threshold_analysis.csv"

THRESHOLDS = [0.3, 0.4, 0.5, 0.6, 0.65, 0.7]


def main() -> None:
    # Cargamos el modelo ya entrenado y los datos completos (con customerID)
    pipeline = joblib.load(MODEL_PATH)
    df = load_customers()

    # Reproducimos el mismo split 80/20 que usó train.py (mismo random_state)
    # para analizar solo clientes que el modelo no vio al entrenar.
    X_train, X_test, y_train, y_test = split_data(df)

    # X_test no tiene customerID (se dropeó en split_data), pero conserva el
    # índice original del df, así que lo recuperamos desde ahí.
    customer_ids_test = df.loc[X_test.index, "customerID"]

    y_pred = pipeline.predict(X_test)
    y_proba = pipeline.predict_proba(X_test)[:, 1]

    # --- Coeficientes con nombres reales de columnas ---
    preprocessor = pipeline.named_steps["preprocessor"]
    classifier = pipeline.named_steps["classifier"]
    feature_names = preprocessor.get_feature_names_out()

    print(f"Columnas después del preprocesamiento: {len(feature_names)}")

    coef_df = pd.DataFrame(
        {"feature": feature_names, "coefficient": classifier.coef_[0]}
    ).sort_values("coefficient", ascending=False)

    Path(COEFFICIENTS_PATH).parent.mkdir(parents=True, exist_ok=True)
    coef_df.to_csv(COEFFICIENTS_PATH, index=False)

    print("\nTop 10 features que más aumentan el riesgo de churn:")
    print(coef_df.head(10).to_string(index=False))

    print("\nTop 10 features que más reducen el riesgo de churn:")
    print(coef_df.sort_values("coefficient").head(10).to_string(index=False))

    # --- Predicciones sobre el test set, ordenadas por probabilidad ---
    predictions_df = pd.DataFrame(
        {
            "customerID": customer_ids_test.values,
            "churn_real": y_test.values,
            "churn_proba": y_proba,
            "prediccion": y_pred,
        }
    ).sort_values("churn_proba", ascending=False)

    Path(PREDICTIONS_PATH).parent.mkdir(parents=True, exist_ok=True)
    predictions_df.to_csv(PREDICTIONS_PATH, index=False)

    # --- Análisis de distintos umbrales de decisión ---
    threshold_rows = []
    for threshold in THRESHOLDS:
        y_pred_threshold = (y_proba >= threshold).astype(int)
        threshold_rows.append(
            {
                "threshold": threshold,
                "clientes_marcados_riesgo": int(y_pred_threshold.sum()),
                "precision": precision_score(y_test, y_pred_threshold, zero_division=0),
                "recall": recall_score(y_test, y_pred_threshold, zero_division=0),
            }
        )
    threshold_df = pd.DataFrame(threshold_rows)

    Path(THRESHOLD_PATH).parent.mkdir(parents=True, exist_ok=True)
    threshold_df.to_csv(THRESHOLD_PATH, index=False)


if __name__ == "__main__":
    main()
