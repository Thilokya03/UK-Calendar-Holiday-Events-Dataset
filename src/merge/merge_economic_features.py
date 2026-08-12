from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"

ECONOMIC_DIR = (
    DATA_DIR
    / "economic"
)

PROCESSED_DIR = (
    DATA_DIR
    / "processed"
)

ECONOMIC_DIR.mkdir(
    parents=True,
    exist_ok=True,
)

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. INPUT FILES
# ============================================================

INDUSTRIAL_PATH = (
    ECONOMIC_DIR
    / "industrial_production.csv"
)

GDP_PATH = (
    ECONOMIC_DIR
    / "monthly_gdp.csv"
)

CPI_PATH = (
    ECONOMIC_DIR
    / "cpi.csv"
)

UNEMPLOYMENT_PATH = (
    ECONOMIC_DIR
    / "unemployment.csv"
)

CALENDAR_PATH = (
    PROCESSED_DIR
    / "basic_calendar.csv"
)


# ============================================================
# 3. OUTPUT FILES
# ============================================================

MONTHLY_OUTPUT_PATH = (
    ECONOMIC_DIR
    / "uk_economic_features_monthly.csv"
)

DAILY_OUTPUT_PATH = (
    PROCESSED_DIR
    / "uk_economic_features_daily.csv"
)


# ============================================================
# 4. ECONOMIC FEATURE DEFINITIONS
# ============================================================

FEATURE_FILES = {
    "industrial_production_index":
        INDUSTRIAL_PATH,

    "gdp_index":
        GDP_PATH,

    "cpi_index":
        CPI_PATH,

    "unemployment_rate":
        UNEMPLOYMENT_PATH,
}


# ============================================================
# 5. HELPERS
# ============================================================

def require_file(
    path: Path,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing file:\n"
            f"{path}\n\n"
            f"Run extract_economic_data.py first."
        )


def load_feature(
    path: Path,
    feature_name: str,
) -> pd.DataFrame:

    require_file(
        path
    )

    df = pd.read_csv(
        path
    )

    required = [
        "date",
        feature_name,
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"'{column}' not found in:\n"
                f"{path}"
            )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    invalid_dates = (
        df["date"]
        .isna()
        .sum()
    )

    if invalid_dates > 0:

        raise ValueError(
            f"{invalid_dates} invalid dates "
            f"in {path}"
        )

    df[
        feature_name
    ] = pd.to_numeric(
        df[
            feature_name
        ],
        errors="coerce",
    )

    # --------------------------------------------------------
    # Convert every timestamp to month key
    # --------------------------------------------------------

    df[
        "year_month"
    ] = (
        df[
            "date"
        ]
        .dt.to_period("M")
        .astype(str)
    )

    # --------------------------------------------------------
    # One record per month
    # --------------------------------------------------------

    df = (
        df[
            [
                "year_month",
                feature_name,
            ]
        ]
        .drop_duplicates(
            subset=[
                "year_month"
            ],
            keep="last",
        )
        .sort_values(
            "year_month"
        )
        .reset_index(
            drop=True
        )
    )

    return df


# ============================================================
# 6. LOAD ALL ECONOMIC FEATURES
# ============================================================

def load_all_features():

    print(
        "\n======================================"
    )

    print(
        "1. LOADING ECONOMIC FEATURES"
    )

    print(
        "======================================"
    )

    datasets = {}

    for (
        feature_name,
        path
    ) in FEATURE_FILES.items():

        df = load_feature(
            path=path,
            feature_name=feature_name,
        )

        datasets[
            feature_name
        ] = df

        print(
            f"{feature_name}: "
            f"{len(df):,} months"
        )

    return datasets


# ============================================================
# 7. MERGE MONTHLY FEATURES
# ============================================================

def merge_monthly_features(
    datasets: dict,
) -> pd.DataFrame:

    print(
        "\n======================================"
    )

    print(
        "2. MERGING MONTHLY ECONOMIC FEATURES"
    )

    print(
        "======================================"
    )

    feature_names = list(
        datasets.keys()
    )

    economic = (
        datasets[
            feature_names[0]
        ]
        .copy()
    )

    for feature_name in feature_names[1:]:

        economic = economic.merge(
            datasets[
                feature_name
            ],
            on="year_month",
            how="outer",
            validate="one_to_one",
        )

    economic = (
        economic
        .sort_values(
            "year_month"
        )
        .reset_index(
            drop=True
        )
    )

    # --------------------------------------------------------
    # Month date = first day of observation month
    # --------------------------------------------------------

    economic[
        "date"
    ] = pd.to_datetime(
        economic[
            "year_month"
        ]
        + "-01",
        errors="coerce",
    )

    economic[
        "year"
    ] = (
        economic[
            "date"
        ]
        .dt.year
    )

    economic[
        "month"
    ] = (
        economic[
            "date"
        ]
        .dt.month
    )

    economic = economic[
        [
            "date",
            "year",
            "month",
            "year_month",
            "industrial_production_index",
            "gdp_index",
            "cpi_index",
            "unemployment_rate",
        ]
    ]

    return economic


# ============================================================
# 8. MONTHLY VALIDATION
# ============================================================

def validate_monthly(
    df: pd.DataFrame,
) -> None:

    print(
        "\n======================================"
    )

    print(
        "3. MONTHLY VALIDATION"
    )

    print(
        "======================================"
    )

    duplicates = (
        df[
            "year_month"
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate months:",
        duplicates,
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate economic months exist."
        )

    feature_columns = [
        "industrial_production_index",
        "gdp_index",
        "cpi_index",
        "unemployment_rate",
    ]

    print(
        "\nLatest available observation:"
    )

    for feature in feature_columns:

        valid = df[
            df[
                feature
            ].notna()
        ]

        if valid.empty:

            print(
                f"{feature}: NO DATA"
            )

            continue

        print(
            feature,
            "->",
            valid[
                "date"
            ].max().date(),
        )

    print(
        "\nMissing values:"
    )

    print(
        df[
            feature_columns
        ]
        .isna()
        .sum()
    )


# ============================================================
# 9. SAVE ORIGINAL MONTHLY DATA
# ============================================================

def save_monthly(
    df: pd.DataFrame,
) -> None:

    output = df.copy()

    output[
        "date"
    ] = (
        output[
            "date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output.to_csv(
        MONTHLY_OUTPUT_PATH,
        index=False,
    )

    print(
        "\nMonthly economic data saved:"
    )

    print(
        MONTHLY_OUTPUT_PATH.resolve()
    )


# ============================================================
# 10. CREATE ML-SAFE MONTHLY FEATURES
# ============================================================

def create_lagged_monthly_features(
    monthly: pd.DataFrame,
) -> pd.DataFrame:
    """
    IMPORTANT:

    Monthly statistics are generally not known
    on the first day of the month they describe.

    Therefore, using the same month's value to
    forecast that month's electricity demand can
    introduce future-information leakage.

    For this first safe implementation:

        April demand
            uses
        March economic information

    Later, actual ONS publication dates can be
    used for even more accurate point-in-time data.
    """

    lagged = monthly.copy()

    lagged = (
        lagged
        .sort_values("date")
        .reset_index(drop=True)
    )

    source_features = [
        "industrial_production_index",
        "gdp_index",
        "cpi_index",
        "unemployment_rate",
    ]

    # --------------------------------------------------------
    # Shift every economic variable by one month
    # --------------------------------------------------------

    for feature in source_features:

        lagged[
            f"{feature}_lag1m"
        ] = (
            lagged[
                feature
            ]
            .shift(1)
        )

    # --------------------------------------------------------
    # Previous month identifier
    # --------------------------------------------------------

    lagged[
        "economic_reference_month"
    ] = (
        lagged[
            "date"
        ]
        .dt.to_period("M")
        .shift(1)
        .astype("string")
    )

    return lagged[
        [
            "date",
            "year",
            "month",
            "year_month",
            "economic_reference_month",
            "industrial_production_index_lag1m",
            "gdp_index_lag1m",
            "cpi_index_lag1m",
            "unemployment_rate_lag1m",
        ]
    ]


# ============================================================
# 11. CREATE DAILY ECONOMIC DATA
# ============================================================

def create_daily_features(
    lagged_monthly: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\n======================================"
    )

    print(
        "4. CREATING DAILY ECONOMIC FEATURES"
    )

    print(
        "======================================"
    )

    require_file(
        CALENDAR_PATH
    )

    calendar = pd.read_csv(
        CALENDAR_PATH
    )

    if "date" not in calendar.columns:

        raise ValueError(
            "'date' missing from "
            "basic_calendar.csv"
        )

    calendar[
        "date"
    ] = pd.to_datetime(
        calendar[
            "date"
        ],
        errors="coerce",
    )

    calendar = calendar.dropna(
        subset=[
            "date"
        ]
    )

    if calendar[
        "date"
    ].duplicated().any():

        raise ValueError(
            "Duplicate dates found "
            "in basic_calendar.csv"
        )

    # --------------------------------------------------------
    # Monthly key for calendar
    # --------------------------------------------------------

    calendar[
        "year_month"
    ] = (
        calendar[
            "date"
        ]
        .dt.to_period("M")
        .astype(str)
    )

    economic_features = (
        lagged_monthly[
            [
                "year_month",
                "economic_reference_month",
                "industrial_production_index_lag1m",
                "gdp_index_lag1m",
                "cpi_index_lag1m",
                "unemployment_rate_lag1m",
            ]
        ]
        .copy()
    )

    if economic_features[
        "year_month"
    ].duplicated().any():

        raise ValueError(
            "Duplicate year_month values "
            "in economic dataset."
        )

    # --------------------------------------------------------
    # Merge monthly features into every date
    # --------------------------------------------------------

    daily = calendar[
        [
            "date",
            "year_month",
        ]
    ].merge(
        economic_features,
        on="year_month",
        how="left",
        validate="many_to_one",
    )

    daily[
        "year"
    ] = (
        daily[
            "date"
        ]
        .dt.year
    )

    daily[
        "month"
    ] = (
        daily[
            "date"
        ]
        .dt.month
    )

    # ========================================================
    # Data-completeness flag
    # ========================================================

    feature_columns = [
        "industrial_production_index_lag1m",
        "gdp_index_lag1m",
        "cpi_index_lag1m",
        "unemployment_rate_lag1m",
    ]

    daily[
        "economic_data_complete"
    ] = (
        daily[
            feature_columns
        ]
        .notna()
        .all(axis=1)
        .astype(int)
    )

    daily = daily[
        [
            "date",
            "year",
            "month",
            "year_month",
            "economic_reference_month",
            "industrial_production_index_lag1m",
            "gdp_index_lag1m",
            "cpi_index_lag1m",
            "unemployment_rate_lag1m",
            "economic_data_complete",
        ]
    ]

    return daily


# ============================================================
# 12. DAILY VALIDATION
# ============================================================

def validate_daily(
    daily: pd.DataFrame,
) -> None:

    print(
        "\n======================================"
    )

    print(
        "5. DAILY VALIDATION"
    )

    print(
        "======================================"
    )

    duplicates = (
        daily[
            "date"
        ]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate dates:",
        duplicates,
    )

    if duplicates > 0:

        raise ValueError(
            "Duplicate dates found "
            "in economic daily dataset."
        )

    feature_columns = [
        "industrial_production_index_lag1m",
        "gdp_index_lag1m",
        "cpi_index_lag1m",
        "unemployment_rate_lag1m",
    ]

    print(
        "\nMissing values:"
    )

    print(
        daily[
            feature_columns
        ]
        .isna()
        .sum()
    )

    complete_days = int(
        daily[
            "economic_data_complete"
        ].sum()
    )

    incomplete_days = (
        len(daily)
        - complete_days
    )

    print(
        "\nComplete economic days:",
        complete_days,
    )

    print(
        "Incomplete economic days:",
        incomplete_days,
    )

    # --------------------------------------------------------
    # Latest date where all features are available
    # --------------------------------------------------------

    complete = daily[
        daily[
            "economic_data_complete"
        ] == 1
    ]

    if not complete.empty:

        print(
            "\nLatest complete economic date:",
            complete[
                "date"
            ].max().date(),
        )


# ============================================================
# 13. SAVE DAILY DATA
# ============================================================

def save_daily(
    daily: pd.DataFrame,
) -> None:

    output = daily.copy()

    output[
        "date"
    ] = (
        output[
            "date"
        ]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output.to_csv(
        DAILY_OUTPUT_PATH,
        index=False,
    )

    print(
        "\n======================================"
    )

    print(
        "DAILY ECONOMIC DATA SAVED"
    )

    print(
        "======================================"
    )

    print(
        DAILY_OUTPUT_PATH.resolve()
    )

    print(
        "Shape:",
        output.shape,
    )


# ============================================================
# 14. MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "UK ECONOMIC FEATURE MERGE PIPELINE"
    )

    print(
        "========================================"
    )

    print(
        "Project root:",
        BASE_DIR,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    datasets = (
        load_all_features()
    )

    # --------------------------------------------------------
    # Merge raw monthly observations
    # --------------------------------------------------------

    monthly = (
        merge_monthly_features(
            datasets
        )
    )

    validate_monthly(
        monthly
    )

    save_monthly(
        monthly
    )

    # --------------------------------------------------------
    # Create ML-safe 1-month-lagged version
    # --------------------------------------------------------

    lagged_monthly = (
        create_lagged_monthly_features(
            monthly
        )
    )

    # --------------------------------------------------------
    # Expand monthly values onto daily calendar
    # --------------------------------------------------------

    daily = (
        create_daily_features(
            lagged_monthly
        )
    )

    validate_daily(
        daily
    )

    save_daily(
        daily
    )

    print(
        "\n========================================"
    )

    print(
        "ECONOMIC PIPELINE COMPLETED"
    )

    print(
        "========================================"
    )


if __name__ == "__main__":
    main()