"""Utilities for loading and cleaning price data."""
from __future__ import annotations

from typing import List

import pandas as pd

# Mandatory OHLCV columns that must be present in the CSV.
MANDATORY_COLUMNS: List[str] = ["open", "high", "low", "close", "volume"]
# Optional columns that are allowed to appear in addition to the mandatory ones.
OPTIONAL_COLUMNS: List[str] = ["adjclose"]


def _has_ticker_in_colname(df: pd.DataFrame) -> bool:
    """Return ``True`` if the frame's column index name hints at a ticker column."""
    name = getattr(df.columns, "name", None)
    return isinstance(name, str) and ("Ticker" in name)


def _has_ticker_row(df: pd.DataFrame, *, sample_size: int = 5) -> bool:
    """Return ``True`` if any of the first ``sample_size`` index entries equals ``Ticker``."""
    head_idx = df.index[:sample_size].astype(str).tolist()
    return any(s.strip().lower() == "ticker" for s in head_idx)


def read_prices(csv_path: str) -> pd.DataFrame:
    """Read OHLCV price data from ``csv_path`` into a cleaned ``DataFrame``.

    The CSV is expected to contain the mandatory columns defined in
    :data:`MANDATORY_COLUMNS`. Optional price related columns like ``adjclose``
    listed in :data:`OPTIONAL_COLUMNS` are preserved when present but are not
    required to successfully load the data.

    Args:
        csv_path: Path to the CSV file that contains price data.

    Returns:
        A ``DataFrame`` with a ``DatetimeIndex`` and numeric price columns.

    Raises:
        ValueError: If any of the mandatory columns are missing from the file.
    """
    df0 = pd.read_csv(csv_path, index_col=0)

    has_multi = isinstance(df0.columns, pd.MultiIndex)
    has_ticker_col = _has_ticker_in_colname(df0)
    has_ticker_row = _has_ticker_row(df0)
    has_ticker_in_cols = "Ticker" in df0.columns.tolist()

    if has_multi or has_ticker_col or has_ticker_row or has_ticker_in_cols:
        df1 = pd.read_csv(csv_path, index_col=0, header=[0, 1])
        df1.columns = df1.columns.get_level_values(0)
        df = df1
    else:
        df = df0

    df.index = pd.to_datetime(df.index, errors="coerce")
    df = df[~df.index.isna()]

    missing_required = [c for c in MANDATORY_COLUMNS if c not in df.columns]
    if missing_required:
        missing_list = ", ".join(missing_required)
        raise ValueError(f"Missing required columns: {missing_list}")

    optional_present = [c for c in OPTIONAL_COLUMNS if c in df.columns]
    keep_columns = MANDATORY_COLUMNS + optional_present
    df = df[keep_columns].copy()

    for column in df.columns:
        df[column] = pd.to_numeric(df[column], errors="coerce")

    df = df.dropna(how="any")

    return df


__all__ = ["read_prices", "MANDATORY_COLUMNS", "OPTIONAL_COLUMNS"]
