"""Carga data/processed/telco_clean.csv a una base DuckDB."""

import duckdb
import pandas as pd

CLEAN_PATH = "data/processed/telco_clean.csv"
DB_PATH = "data/warehouse/telco.duckdb"

EXPECTED_ROWS = 7043


def main() -> None:
    df = pd.read_csv(CLEAN_PATH)
    csv_churn_rate = df["Churn"].mean()

    con = duckdb.connect(DB_PATH)
    con.execute("CREATE OR REPLACE TABLE customers AS SELECT * FROM df")

    n_rows = con.execute("SELECT COUNT(*) FROM customers").fetchone()[0]
    db_churn_rate = con.execute("SELECT AVG(Churn) FROM customers").fetchone()[0]

    assert n_rows == EXPECTED_ROWS, (
        f"Se esperaban {EXPECTED_ROWS} filas en customers, hay {n_rows}"
    )
    assert round(db_churn_rate, 4) == round(csv_churn_rate, 4), (
        f"Tasa de churn en DB ({db_churn_rate:.4f}) no coincide con el CSV "
        f"({csv_churn_rate:.4f})"
    )

    print(f"Filas cargadas: {n_rows}")

    print("\nTasa de churn por tipo de Contract:")
    result = con.execute(
        """
        SELECT Contract, AVG(Churn) AS churn_rate
        FROM customers
        GROUP BY Contract
        ORDER BY churn_rate DESC
        """
    ).fetchdf()
    print(result.to_string(index=False))

    con.close()


if __name__ == "__main__":
    main()
