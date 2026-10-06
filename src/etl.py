"""ETL: limpieza del dataset crudo de Telco Customer Churn."""

import pandas as pd

RAW_PATH = "data/raw/Telco-Customer-Churn.csv"
PROCESSED_PATH = "data/processed/telco_clean.csv"


def load_raw(path: str) -> pd.DataFrame:
    return pd.read_csv(path)


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    total_charges = pd.to_numeric(df["TotalCharges"], errors="coerce")
    missing_mask = total_charges.isna()
    assert (df.loc[missing_mask, "tenure"] == 0).all(), (
        "Hay filas con TotalCharges vacío cuyo tenure no es 0"
    )
    df["TotalCharges"] = total_charges.fillna(0)

    df["Churn"] = df["Churn"].map({"Yes": 1, "No": 0})

    return df


def main() -> None:
    df = load_raw(RAW_PATH)
    df_clean = clean(df)
    df_clean.to_csv(PROCESSED_PATH, index=False)

    n_rows = len(df_clean)
    churn_rate = df_clean["Churn"].mean()
    print(f"Filas procesadas: {n_rows}")
    print(f"Tasa de churn global: {churn_rate:.2%}")


if __name__ == "__main__":
    main()
