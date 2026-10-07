import pandas as pd
import pytest

from etl import clean


def test_total_charges_empty_with_tenure_zero_filled_with_zero():
    df = pd.DataFrame(
        {
            "TotalCharges": [" "],
            "tenure": [0],
            "Churn": ["No"],
        }
    )

    result = clean(df)

    assert result.loc[0, "TotalCharges"] == 0


def test_total_charges_empty_with_tenure_nonzero_raises():
    df = pd.DataFrame(
        {
            "TotalCharges": [" "],
            "tenure": [5],
            "Churn": ["No"],
        }
    )

    with pytest.raises(AssertionError):
        clean(df)


def test_churn_yes_no_mapped_to_1_0():
    df = pd.DataFrame(
        {
            "TotalCharges": ["10.5", "20.0"],
            "tenure": [1, 2],
            "Churn": ["Yes", "No"],
        }
    )

    result = clean(df)

    assert result["Churn"].tolist() == [1, 0]


def test_clean_does_not_mutate_original():
    df = pd.DataFrame(
        {
            "TotalCharges": [" "],
            "tenure": [0],
            "Churn": ["Yes"],
        }
    )

    clean(df)

    assert df.loc[0, "TotalCharges"] == " "
    assert df.loc[0, "Churn"] == "Yes"
