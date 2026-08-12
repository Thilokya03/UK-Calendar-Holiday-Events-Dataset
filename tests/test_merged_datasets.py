from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_series_equal


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

FULL_CALENDAR_PATH = Path(
    os.environ.get(
        "FULL_CALENDAR_PATH",
        PROCESSED_DIR / "full_calendar_features.csv",
    )
)
ECONOMY_DATA_PATH = Path(
    os.environ.get(
        "ECONOMY_DATA_PATH",
        PROCESSED_DIR / "uk_economic_features_daily.csv",
    )
)


CALENDAR_REQUIRED_COLUMNS = {
    "date",
    "year",
    "month_num",
    "month_name",
    "day",
    "day_of_week",
    "day_of_week_num",
    "week_of_year",
    "quarter",
    "is_weekend",
    "season",
    "holiday_names",
    "holiday_regions",
    "is_bank_holiday_england_wales",
    "is_bank_holiday_scotland",
    "is_bank_holiday",
    "is_non_working_day",
    "is_event_day",
    "event_count",
    "event_names",
}

ECONOMY_FEATURE_COLUMNS = [
    "industrial_production_index_lag1m",
    "gdp_index_lag1m",
    "cpi_index_lag1m",
    "unemployment_rate_lag1m",
]

ECONOMY_REQUIRED_COLUMNS = {
    "date",
    "year",
    "month",
    "year_month",
    "economic_reference_month",
    *ECONOMY_FEATURE_COLUMNS,
    "economic_data_complete",
}

SEASON_BY_MONTH = {
    1: "Winter",
    2: "Winter",
    3: "Spring",
    4: "Spring",
    5: "Spring",
    6: "Summer",
    7: "Summer",
    8: "Summer",
    9: "Autumn",
    10: "Autumn",
    11: "Autumn",
    12: "Winter",
}


def _read_dataset(path: Path) -> pd.DataFrame:
    """Read a required CSV and parse its date without silently losing rows."""
    assert path.exists(), f"Dataset does not exist: {path}"
    assert path.is_file(), f"Dataset path is not a file: {path}"
    assert path.stat().st_size > 0, f"Dataset is empty: {path}"

    df = pd.read_csv(path, keep_default_na=True)
    assert not df.empty, f"Dataset has no data rows: {path}"
    assert "date" in df.columns, f"Missing 'date' column in {path.name}"

    parsed_dates = pd.to_datetime(df["date"], errors="coerce")
    invalid = df.loc[parsed_dates.isna(), "date"].head(10).tolist()
    assert parsed_dates.notna().all(), (
        f"Invalid date values in {path.name}; examples: {invalid}"
    )

    df["date"] = parsed_dates
    return df


def _assert_required_columns(
    df: pd.DataFrame,
    required: set[str],
    dataset_name: str,
) -> None:
    missing = sorted(required - set(df.columns))
    assert not missing, f"Missing columns in {dataset_name}: {missing}"


def _assert_daily_date_index(df: pd.DataFrame, dataset_name: str) -> None:
    duplicates = df.loc[df["date"].duplicated(keep=False), "date"]
    assert duplicates.empty, (
        f"Duplicate dates in {dataset_name}: "
        f"{duplicates.dt.strftime('%Y-%m-%d').head(10).tolist()}"
    )

    assert df["date"].is_monotonic_increasing, (
        f"Dates are not sorted in ascending order in {dataset_name}"
    )

    expected = pd.date_range(df["date"].min(), df["date"].max(), freq="D")
    missing_dates = expected.difference(pd.DatetimeIndex(df["date"]))
    assert missing_dates.empty, (
        f"Missing calendar dates in {dataset_name}; first missing dates: "
        f"{missing_dates[:10].strftime('%Y-%m-%d').tolist()}"
    )
    assert len(df) == len(expected), (
        f"Expected {len(expected)} daily rows in {dataset_name}, found {len(df)}"
    )


def _assert_binary_columns(
    df: pd.DataFrame,
    columns: list[str],
    dataset_name: str,
) -> None:
    for column in columns:
        assert not df[column].isna().any(), (
            f"'{column}' contains missing values in {dataset_name}"
        )
        numeric = pd.to_numeric(df[column], errors="coerce")
        assert numeric.notna().all(), (
            f"'{column}' contains non-numeric values in {dataset_name}"
        )
        invalid = sorted(set(numeric.dropna().unique()) - {0, 1})
        assert not invalid, (
            f"'{column}' must contain only 0/1 in {dataset_name}; "
            f"invalid values: {invalid[:10]}"
        )


@pytest.fixture(scope="module")
def calendar_df() -> pd.DataFrame:
    return _read_dataset(FULL_CALENDAR_PATH)


@pytest.fixture(scope="module")
def economy_df() -> pd.DataFrame:
    return _read_dataset(ECONOMY_DATA_PATH)


def test_full_calendar_schema(calendar_df: pd.DataFrame) -> None:
    _assert_required_columns(
        calendar_df,
        CALENDAR_REQUIRED_COLUMNS,
        FULL_CALENDAR_PATH.name,
    )


def test_full_calendar_has_one_row_per_day(calendar_df: pd.DataFrame) -> None:
    _assert_daily_date_index(calendar_df, FULL_CALENDAR_PATH.name)


def test_full_calendar_derived_date_columns(calendar_df: pd.DataFrame) -> None:
    dates = calendar_df["date"]

    expected = {
        "year": dates.dt.year,
        "month_num": dates.dt.month,
        "month_name": dates.dt.month_name(),
        "day": dates.dt.day,
        "day_of_week": dates.dt.day_name(),
        "day_of_week_num": dates.dt.dayofweek,
        "week_of_year": dates.dt.isocalendar().week.astype("int64"),
        "quarter": dates.dt.quarter,
        "season": dates.dt.month.map(SEASON_BY_MONTH),
    }

    for column, expected_values in expected.items():
        actual = calendar_df[column]
        if column not in {"month_name", "day_of_week", "season"}:
            actual = pd.to_numeric(actual, errors="coerce")
        assert_series_equal(
            actual.reset_index(drop=True),
            expected_values.reset_index(drop=True),
            check_dtype=False,
            check_names=False,
            obj=column,
        )


def test_full_calendar_binary_flags(calendar_df: pd.DataFrame) -> None:
    flag_columns = [
        column
        for column in calendar_df.columns
        if column.startswith("is_")
    ]
    assert flag_columns, "No binary 'is_' feature columns were found"
    _assert_binary_columns(calendar_df, flag_columns, FULL_CALENDAR_PATH.name)


def test_weekend_flag_is_correct(calendar_df: pd.DataFrame) -> None:
    expected = (calendar_df["date"].dt.dayofweek >= 5).astype(int)
    actual = pd.to_numeric(calendar_df["is_weekend"], errors="coerce")
    assert_series_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )


def test_bank_holiday_relationships(calendar_df: pd.DataFrame) -> None:
    overall = pd.to_numeric(calendar_df["is_bank_holiday"])
    england_wales = pd.to_numeric(
        calendar_df["is_bank_holiday_england_wales"]
    )
    scotland = pd.to_numeric(calendar_df["is_bank_holiday_scotland"])

    assert (overall >= england_wales).all(), (
        "England/Wales holiday found where is_bank_holiday is 0"
    )
    assert (overall >= scotland).all(), (
        "Scotland holiday found where is_bank_holiday is 0"
    )
    assert overall.sum() > 0, "No bank holidays are marked in the dataset"


def test_non_working_day_rule(calendar_df: pd.DataFrame) -> None:
    weekend = pd.to_numeric(calendar_df["is_weekend"])
    holiday = pd.to_numeric(calendar_df["is_bank_holiday"])
    expected = ((weekend == 1) | (holiday == 1)).astype(int)
    actual = pd.to_numeric(calendar_df["is_non_working_day"])

    assert_series_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )


def test_event_count_and_event_day_agree(calendar_df: pd.DataFrame) -> None:
    event_count = pd.to_numeric(calendar_df["event_count"], errors="coerce")
    assert event_count.notna().all(), "event_count contains missing/non-numeric values"
    assert (event_count >= 0).all(), "event_count contains negative values"
    assert (event_count % 1 == 0).all(), "event_count contains non-integer values"

    expected = (event_count > 0).astype(int)
    actual = pd.to_numeric(calendar_df["is_event_day"])
    assert_series_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )


def test_economy_schema(economy_df: pd.DataFrame) -> None:
    _assert_required_columns(
        economy_df,
        ECONOMY_REQUIRED_COLUMNS,
        ECONOMY_DATA_PATH.name,
    )


def test_economy_has_one_row_per_day(economy_df: pd.DataFrame) -> None:
    _assert_daily_date_index(economy_df, ECONOMY_DATA_PATH.name)


def test_economy_calendar_keys(economy_df: pd.DataFrame) -> None:
    dates = economy_df["date"]
    expected_year = dates.dt.year
    expected_month = dates.dt.month
    expected_year_month = dates.dt.to_period("M").astype(str)

    assert_series_equal(
        pd.to_numeric(economy_df["year"], errors="coerce").reset_index(drop=True),
        expected_year.reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )
    assert_series_equal(
        pd.to_numeric(economy_df["month"], errors="coerce").reset_index(drop=True),
        expected_month.reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )
    assert_series_equal(
        economy_df["year_month"].astype(str).reset_index(drop=True),
        pd.Series(expected_year_month).reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )


def test_economy_features_are_numeric(economy_df: pd.DataFrame) -> None:
    for column in ECONOMY_FEATURE_COLUMNS:
        original_non_null = economy_df[column].notna()
        numeric = pd.to_numeric(economy_df[column], errors="coerce")
        newly_invalid = original_non_null & numeric.isna()
        assert not newly_invalid.any(), (
            f"'{column}' contains non-numeric values; examples: "
            f"{economy_df.loc[newly_invalid, column].head(10).tolist()}"
        )
        assert numeric.notna().any(), f"'{column}' has no usable numeric values"


def test_economic_completeness_flag(economy_df: pd.DataFrame) -> None:
    _assert_binary_columns(
        economy_df,
        ["economic_data_complete"],
        ECONOMY_DATA_PATH.name,
    )
    numeric_features = economy_df[ECONOMY_FEATURE_COLUMNS].apply(
        pd.to_numeric,
        errors="coerce",
    )
    expected = numeric_features.notna().all(axis=1).astype(int)
    actual = pd.to_numeric(economy_df["economic_data_complete"])

    assert_series_equal(
        actual.reset_index(drop=True),
        expected.reset_index(drop=True),
        check_dtype=False,
        check_names=False,
    )


def test_economic_reference_month_has_no_future_leakage(
    economy_df: pd.DataFrame,
) -> None:
    available = economy_df["economic_reference_month"].notna()
    references = economy_df.loc[available, "economic_reference_month"].astype(str)
    parsed_references = pd.to_datetime(references + "-01", errors="coerce")

    assert parsed_references.notna().all(), (
        "economic_reference_month contains invalid YYYY-MM values"
    )
    current_months = economy_df.loc[available, "date"].dt.to_period("M")
    reference_months = parsed_references.dt.to_period("M")
    assert (reference_months < current_months).all(), (
        "Economic reference month must be earlier than the demand/calendar month"
    )


def test_daily_economic_values_are_constant_within_each_month(
    economy_df: pd.DataFrame,
) -> None:
    columns = ["economic_reference_month", *ECONOMY_FEATURE_COLUMNS]
    unique_counts = economy_df.groupby("year_month", dropna=False)[columns].nunique(
        dropna=False
    )
    changing = unique_counts.gt(1).any(axis=1)
    assert not changing.any(), (
        "Daily economic values change within these months: "
        f"{changing[changing].index[:10].tolist()}"
    )


def test_calendar_and_economy_dates_match(
    calendar_df: pd.DataFrame,
    economy_df: pd.DataFrame,
) -> None:
    calendar_dates = pd.DatetimeIndex(calendar_df["date"])
    economy_dates = pd.DatetimeIndex(economy_df["date"])

    missing_in_economy = calendar_dates.difference(economy_dates)
    extra_in_economy = economy_dates.difference(calendar_dates)
    assert missing_in_economy.empty and extra_in_economy.empty, (
        "Calendar/economy date coverage differs. "
        f"Missing in economy: {missing_in_economy[:5].strftime('%Y-%m-%d').tolist()}; "
        f"extra in economy: {extra_in_economy[:5].strftime('%Y-%m-%d').tolist()}"
    )