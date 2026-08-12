from __future__ import annotations

from pathlib import Path

import pandas as pd


# ============================================================
# 1. PATH CONFIGURATION
# ============================================================

BASE_DIR = Path(__file__).resolve().parents[2]

DATA_DIR = BASE_DIR / "data"
PROCESSED_DIR = DATA_DIR / "processed"

PROCESSED_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ============================================================
# 2. INPUT FILES
# ============================================================

BASIC_CALENDAR_PATH = (
    PROCESSED_DIR
    / "basic_calendar.csv"
)

CURRENT_HOLIDAY_PATH = (
    PROCESSED_DIR
    / "combined_holidays.csv"
)

HISTORICAL_HOLIDAY_PATH = (
    PROCESSED_DIR
    / "dmo_historical_holidays.csv"
)

EVENT_FEATURE_PATH = (
    PROCESSED_DIR
    / "event_daily_features.csv"
)


# ============================================================
# 3. OUTPUT FILE
# ============================================================

OUTPUT_PATH = (
    PROCESSED_DIR
    / "full_calendar_features.csv"
)


# ============================================================
# 4. HELPERS
# ============================================================

def require_file(
    path: Path,
    name: str,
) -> None:

    if not path.exists():

        raise FileNotFoundError(
            f"\nMissing {name}:\n"
            f"{path}\n"
        )


def load_csv_with_date(
    path: Path,
) -> pd.DataFrame:

    df = pd.read_csv(
        path
    )

    if "date" not in df.columns:

        raise ValueError(
            f"'date' column is missing from:\n"
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
            f"found in {path}"
        )

    return df


def join_unique_text(
    values,
) -> str:

    cleaned = {
        str(value).strip()
        for value in values
        if (
            pd.notna(value)
            and str(value).strip()
        )
    }

    return " | ".join(
        sorted(cleaned)
    )


# ============================================================
# 5. BASIC CALENDAR
# ============================================================

def load_basic_calendar() -> pd.DataFrame:

    print(
        "\n======================================"
    )

    print(
        "1. BASIC CALENDAR"
    )

    print(
        "======================================"
    )

    require_file(
        BASIC_CALENDAR_PATH,
        "basic calendar",
    )

    calendar = load_csv_with_date(
        BASIC_CALENDAR_PATH
    )

    if calendar[
        "date"
    ].duplicated().any():

        raise ValueError(
            "Duplicate dates exist in "
            "basic_calendar.csv"
        )

    calendar = (
        calendar
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        f"Rows: {len(calendar):,}"
    )

    print(
        "Date range:",
        calendar["date"].min().date(),
        "->",
        calendar["date"].max().date(),
    )

    return calendar


# ============================================================
# 6. CURRENT GOV.UK HOLIDAYS
# ============================================================

def prepare_current_holidays() -> pd.DataFrame:

    print(
        "\n======================================"
    )

    print(
        "2. CURRENT GOV.UK HOLIDAYS"
    )

    print(
        "======================================"
    )

    require_file(
        CURRENT_HOLIDAY_PATH,
        "current holiday dataset",
    )

    df = load_csv_with_date(
        CURRENT_HOLIDAY_PATH
    )

    required = [
        "date",
        "holiday_name",
        "region",
    ]

    for column in required:

        if column not in df.columns:

            raise ValueError(
                f"Missing '{column}' in "
                "combined_holidays.csv"
            )

    # --------------------------------------------------------
    # Clean region names
    # --------------------------------------------------------

    df["region"] = (
        df["region"]
        .astype(str)
        .str.strip()
        .str.lower()
    )

    # --------------------------------------------------------
    # Regional holiday indicators
    # --------------------------------------------------------

    df[
        "is_bank_holiday_england_wales"
    ] = (
        df["region"]
        .isin(
            [
                "england-and-wales",
                "england and wales",
            ]
        )
    ).astype(int)

    df[
        "is_bank_holiday_scotland"
    ] = (
        df["region"]
        .eq("scotland")
    ).astype(int)

    # --------------------------------------------------------
    # Aggregate one row per date
    # --------------------------------------------------------

    daily = (
        df
        .groupby(
            "date",
            as_index=False,
        )
        .agg(
            holiday_names=(
                "holiday_name",
                join_unique_text,
            ),

            holiday_regions=(
                "region",
                join_unique_text,
            ),

            is_bank_holiday_england_wales=(
                "is_bank_holiday_england_wales",
                "max",
            ),

            is_bank_holiday_scotland=(
                "is_bank_holiday_scotland",
                "max",
            ),
        )
    )

    # Internal indicator only.
    daily[
        "_is_current_holiday"
    ] = 1

    print(
        "Current unique holiday dates:",
        len(daily),
    )

    return daily


# ============================================================
# 7. HISTORICAL HOLIDAYS
# ============================================================

def prepare_historical_holidays() -> pd.DataFrame:

    print(
        "\n======================================"
    )

    print(
        "3. HISTORICAL HOLIDAYS"
    )

    print(
        "======================================"
    )

    require_file(
        HISTORICAL_HOLIDAY_PATH,
        "historical holiday dataset",
    )

    df = load_csv_with_date(
        HISTORICAL_HOLIDAY_PATH
    )

    # --------------------------------------------------------
    # We only require date because the historical source
    # may not contain the same regional/name structure
    # as GOV.UK.
    # --------------------------------------------------------

    historical = (
        df[
            ["date"]
        ]
        .drop_duplicates()
        .copy()
    )

    # Internal source indicator.
    historical[
        "_is_historical_holiday"
    ] = 1

    print(
        "Historical holiday dates:",
        len(historical),
    )

    print(
        "Historical range:",
        historical["date"].min().date(),
        "->",
        historical["date"].max().date(),
    )

    return historical


# ============================================================
# 8. COMBINE HOLIDAY SOURCES
# ============================================================

def combine_holidays(
    current: pd.DataFrame,
    historical: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\n======================================"
    )

    print(
        "4. COMBINING HOLIDAY SOURCES"
    )

    print(
        "======================================"
    )

    holidays = current.merge(
        historical,
        on="date",
        how="outer",
        validate="one_to_one",
    )

    # --------------------------------------------------------
    # Fill internal source flags
    # --------------------------------------------------------

    holidays[
        "_is_current_holiday"
    ] = (
        holidays[
            "_is_current_holiday"
        ]
        .fillna(0)
        .astype(int)
    )

    holidays[
        "_is_historical_holiday"
    ] = (
        holidays[
            "_is_historical_holiday"
        ]
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Fill regional flags
    # --------------------------------------------------------

    regional_columns = [
        "is_bank_holiday_england_wales",
        "is_bank_holiday_scotland",
    ]

    for column in regional_columns:

        holidays[column] = (
            holidays[column]
            .fillna(0)
            .astype(int)
        )

    # --------------------------------------------------------
    # IMPORTANT FIX
    #
    # Overall bank holiday:
    #
    # Current GOV.UK regional holiday
    #             OR
    # Historical holiday
    #
    # This fixes the previous all-zero column.
    # --------------------------------------------------------

    holidays[
        "is_bank_holiday"
    ] = (
        (
            holidays[
                "is_bank_holiday_england_wales"
            ] == 1
        )
        |
        (
            holidays[
                "is_bank_holiday_scotland"
            ] == 1
        )
        |
        (
            holidays[
                "_is_historical_holiday"
            ] == 1
        )
    ).astype(int)

    # --------------------------------------------------------
    # Text columns
    # --------------------------------------------------------

    holidays[
        "holiday_names"
    ] = (
        holidays[
            "holiday_names"
        ]
        .fillna("")
    )

    holidays[
        "holiday_regions"
    ] = (
        holidays[
            "holiday_regions"
        ]
        .fillna("")
    )

    # --------------------------------------------------------
    # Remove source-specific flags.
    #
    # These describe WHERE the data came from,
    # not a real-world ML feature.
    # --------------------------------------------------------

    holidays = holidays.drop(
        columns=[
            "_is_current_holiday",
            "_is_historical_holiday",
        ]
    )

    holidays = (
        holidays
        .sort_values("date")
        .reset_index(drop=True)
    )

    print(
        "Total unique holiday dates:",
        int(
            holidays[
                "is_bank_holiday"
            ].sum()
        ),
    )

    return holidays


# ============================================================
# 9. EVENT FEATURES
# ============================================================

def load_event_features() -> pd.DataFrame:

    print(
        "\n======================================"
    )

    print(
        "5. EVENT FEATURES"
    )

    print(
        "======================================"
    )

    require_file(
        EVENT_FEATURE_PATH,
        "event feature dataset",
    )

    events = load_csv_with_date(
        EVENT_FEATURE_PATH
    )

    if events[
        "date"
    ].duplicated().any():

        raise ValueError(
            "Duplicate dates exist in "
            "event_daily_features.csv"
        )

    print(
        "Event feature rows:",
        len(events),
    )

    return events


# ============================================================
# 10. MERGE CALENDAR + HOLIDAYS + EVENTS
# ============================================================

def merge_all_features(
    calendar: pd.DataFrame,
    holidays: pd.DataFrame,
    events: pd.DataFrame,
) -> pd.DataFrame:

    print(
        "\n======================================"
    )

    print(
        "6. MERGING ALL CALENDAR FEATURES"
    )

    print(
        "======================================"
    )

    merged = calendar.merge(
        holidays,
        on="date",
        how="left",
        validate="one_to_one",
    )

    merged = merged.merge(
        events,
        on="date",
        how="left",
        validate="one_to_one",
    )

    # ========================================================
    # Holiday flags
    # ========================================================

    holiday_flags = [
        "is_bank_holiday",
        "is_bank_holiday_england_wales",
        "is_bank_holiday_scotland",
    ]

    for column in holiday_flags:

        if column in merged.columns:

            merged[column] = (
                merged[column]
                .fillna(0)
                .astype(int)
            )

    # ========================================================
    # Holiday text
    # ========================================================

    for column in [
        "holiday_names",
        "holiday_regions",
    ]:

        if column in merged.columns:

            merged[column] = (
                merged[column]
                .fillna("")
            )

    # ========================================================
    # Event flags
    # ========================================================

    event_flag_columns = [
        column
        for column in events.columns
        if column.startswith("is_")
    ]

    for column in event_flag_columns:

        if column in merged.columns:

            merged[column] = (
                merged[column]
                .fillna(0)
                .astype(int)
            )

    # ========================================================
    # Event count
    # ========================================================

    if "event_count" in merged.columns:

        merged[
            "event_count"
        ] = (
            merged[
                "event_count"
            ]
            .fillna(0)
            .astype(int)
        )

    # ========================================================
    # Event names
    # ========================================================

    if "event_names" in merged.columns:

        merged[
            "event_names"
        ] = (
            merged[
                "event_names"
            ]
            .fillna("")
        )

    # ========================================================
    # Weekend
    # ========================================================

    if "is_weekend" in merged.columns:

        merged[
            "is_weekend"
        ] = (
            merged[
                "is_weekend"
            ]
            .astype(int)
        )

    # ========================================================
    # IMPORTANT FIX
    #
    # Non-working day:
    #
    # weekend OR bank holiday
    # ========================================================

    merged[
        "is_non_working_day"
    ] = (
        (
            merged[
                "is_weekend"
            ] == 1
        )
        |
        (
            merged[
                "is_bank_holiday"
            ] == 1
        )
    ).astype(int)

    merged = (
        merged
        .sort_values("date")
        .reset_index(drop=True)
    )

    return merged


# ============================================================
# 11. VALIDATION
# ============================================================

def validate_calendar(
    df: pd.DataFrame,
) -> None:

    print(
        "\n======================================"
    )

    print(
        "7. VALIDATION"
    )

    print(
        "======================================"
    )

    # --------------------------------------------------------
    # Duplicate dates
    # --------------------------------------------------------

    duplicate_dates = (
        df["date"]
        .duplicated()
        .sum()
    )

    print(
        "Duplicate dates:",
        duplicate_dates,
    )

    if duplicate_dates > 0:

        raise ValueError(
            "Duplicate dates found "
            "in final calendar."
        )

    # --------------------------------------------------------
    # Missing calendar dates
    # --------------------------------------------------------

    expected_dates = pd.date_range(
        start=df["date"].min(),
        end=df["date"].max(),
        freq="D",
    )

    actual_dates = set(
        df["date"]
    )

    missing_dates = [
        date
        for date in expected_dates
        if date not in actual_dates
    ]

    print(
        "Missing dates:",
        len(missing_dates),
    )

    if missing_dates:

        raise ValueError(
            "Final calendar has "
            "missing dates."
        )

    # --------------------------------------------------------
    # Holiday checks
    # --------------------------------------------------------

    print(
        "\nOverall bank holidays:",
        int(
            df[
                "is_bank_holiday"
            ].sum()
        ),
    )

    print(
        "England/Wales bank holidays:",
        int(
            df[
                "is_bank_holiday_england_wales"
            ].sum()
        ),
    )

    print(
        "Scotland bank holidays:",
        int(
            df[
                "is_bank_holiday_scotland"
            ].sum()
        ),
    )

    print(
        "Non-working days:",
        int(
            df[
                "is_non_working_day"
            ].sum()
        ),
    )

    # --------------------------------------------------------
    # Important sanity test
    # --------------------------------------------------------

    impossible = df[
        (
            df[
                "is_bank_holiday_england_wales"
            ] == 1
        )
        &
        (
            df[
                "is_bank_holiday"
            ] == 0
        )
    ]

    impossible_2 = df[
        (
            df[
                "is_bank_holiday_scotland"
            ] == 1
        )
        &
        (
            df[
                "is_bank_holiday"
            ] == 0
        )
    ]

    if (
        not impossible.empty
        or not impossible_2.empty
    ):

        raise ValueError(
            "Regional holiday exists "
            "but is_bank_holiday == 0."
        )

    # --------------------------------------------------------
    # Non-working day check
    # --------------------------------------------------------

    invalid_nonworking = df[
        (
            (
                df[
                    "is_weekend"
                ] == 1
            )
            |
            (
                df[
                    "is_bank_holiday"
                ] == 1
            )
        )
        &
        (
            df[
                "is_non_working_day"
            ] == 0
        )
    ]

    if not invalid_nonworking.empty:

        raise ValueError(
            "is_non_working_day "
            "calculation is incorrect."
        )

    print(
        "\nCalendar validation passed."
    )


# ============================================================
# 12. SAVE
# ============================================================

def save_calendar(
    df: pd.DataFrame,
) -> None:

    output = df.copy()

    output["date"] = (
        output["date"]
        .dt.strftime(
            "%Y-%m-%d"
        )
    )

    output.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    print(
        "\n======================================"
    )

    print(
        "CALENDAR DATASET SAVED"
    )

    print(
        "======================================"
    )

    print(
        OUTPUT_PATH.resolve()
    )

    print(
        "Shape:",
        output.shape,
    )


# ============================================================
# 13. MAIN
# ============================================================

def main():

    print(
        "\n========================================"
    )

    print(
        "UK FULL CALENDAR FEATURE PIPELINE"
    )

    print(
        "========================================"
    )

    print(
        "Project root:",
        BASE_DIR,
    )

    calendar = (
        load_basic_calendar()
    )

    current_holidays = (
        prepare_current_holidays()
    )

    historical_holidays = (
        prepare_historical_holidays()
    )

    holidays = (
        combine_holidays(
            current=current_holidays,
            historical=historical_holidays,
        )
    )

    events = (
        load_event_features()
    )

    final_calendar = (
        merge_all_features(
            calendar=calendar,
            holidays=holidays,
            events=events,
        )
    )

    validate_calendar(
        final_calendar
    )

    save_calendar(
        final_calendar
    )


if __name__ == "__main__":
    main()