from pathlib import Path

import pandas as pd
import pytest

from etl import RAW_PATH, clean, load_raw


@pytest.fixture(scope="module")
def clean_df() -> pd.DataFrame:
    if not Path(RAW_PATH).exists():
        pytest.skip(f"{RAW_PATH} no existe, se salta la validación de datos reales")
    return clean(load_raw(RAW_PATH))


def test_row_count(clean_df):
    assert len(clean_df) == 7043


def test_no_nulls(clean_df):
    assert clean_df.isna().sum().sum() == 0


def test_churn_values(clean_df):
    assert set(clean_df["Churn"].unique()) == {0, 1}


def test_customer_id_unique(clean_df):
    assert clean_df["customerID"].is_unique


def test_total_charges_numeric_and_non_negative(clean_df):
    assert pd.api.types.is_numeric_dtype(clean_df["TotalCharges"])
    assert (clean_df["TotalCharges"] >= 0).all()
