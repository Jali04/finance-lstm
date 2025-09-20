import pandas as pd
import pytest

from finance_lstm.data import OPTIONAL_COLUMNS, read_prices


def _write_csv(df: pd.DataFrame, path):
    df.to_csv(path, index_label="date")


def test_read_prices_missing_required_columns(tmp_path):
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "close": [1.5, 2.5],
            "volume": [100, 200],
        },
        index=pd.Index(["2024-01-01", "2024-01-02"], name="date"),
    )
    csv_path = tmp_path / "prices.csv"
    _write_csv(df, csv_path)

    with pytest.raises(ValueError) as excinfo:
        read_prices(csv_path)

    assert "Missing required columns: low" == str(excinfo.value)


def test_read_prices_preserves_optional_columns(tmp_path):
    df = pd.DataFrame(
        {
            "open": [1.0, 2.0],
            "high": [2.0, 3.0],
            "low": [0.5, 1.5],
            "close": [1.5, 2.5],
            "volume": [100, 200],
            OPTIONAL_COLUMNS[0]: [1.2, 2.2],
        },
        index=pd.Index(["2024-01-01", "2024-01-02"], name="date"),
    )
    csv_path = tmp_path / "prices.csv"
    _write_csv(df, csv_path)

    result = read_prices(csv_path)

    assert OPTIONAL_COLUMNS[0] in result.columns
    assert result.shape == df.shape


def test_read_prices_normalises_column_names(tmp_path):
    df = pd.DataFrame(
        {
            " Open ": [1.0, 2.0],
            "HIGH": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Close": [1.5, 2.5],
            "Volume": [100, 200],
        },
        index=pd.Index(["2024-01-01", "2024-01-02"], name="date"),
    )
    csv_path = tmp_path / "prices.csv"
    _write_csv(df, csv_path)

    result = read_prices(csv_path)

    assert set(result.columns) == {
        "open",
        "high",
        "low",
        "close",
        "volume",
    }


def test_read_prices_maps_price_alias_to_close(tmp_path):
    df = pd.DataFrame(
        {
            "Price": [1.5, 2.5],
            "High": [2.0, 3.0],
            "Low": [0.5, 1.5],
            "Open": [1.0, 2.0],
            "Volume": [100, 200],
        },
        index=pd.Index(["2024-01-01", "2024-01-02"], name="date"),
    )
    csv_path = tmp_path / "prices.csv"
    _write_csv(df, csv_path)

    result = read_prices(csv_path)

    assert "close" in result.columns
    # Ensure the alias did not lead to duplicate columns being kept.
    assert set(result.columns) == {"open", "high", "low", "close", "volume"}
