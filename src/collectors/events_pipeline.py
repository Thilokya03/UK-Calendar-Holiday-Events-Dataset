from __future__ import annotations

import re
from io import StringIO
from pathlib import Path
from datetime import datetime

import pandas as pd
import requests


# ============================================================
# 1. CONFIGURATION
# ============================================================

START_DATE = pd.Timestamp("2000-01-01")
END_DATE = pd.Timestamp.today().normalize()

BASE_DIR = Path(__file__).resolve().parents[2]

RAW_DIR = BASE_DIR / "data" / "raw" / "events"
PROCESSED_DIR = BASE_DIR / "data" / "processed"

RAW_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


# ------------------------------------------------------------
# Source URLs
# ------------------------------------------------------------

PARLIAMENT_ELECTION_URL = (
    "https://commonslibrary.parliament.uk/research-briefings/sn04512/"
)

FOOTBALL_RESULTS_URL = (
    "https://raw.githubusercontent.com/"
    "martj42/international_results/master/results.csv"
)

ENGLAND_FOOTBALL_VERIFICATION_URL = (
    "https://www.englandfootball.com/"
    "england/mens-senior-team/fixtures-results"
)


# ============================================================
# 2. STANDARD EVENT SCHEMA
# ============================================================

EVENT_COLUMNS = [
    "event_id",
    "event_name",
    "event_type",
    "start_date",
    "end_date",
    "start_time",
    "end_time",
    "all_day",
    "scope",
    "importance",
    "availability",
    "use_for_forecast",
    "source_type",
    "source_name",
    "source_url",
    "notes",
]


# ============================================================
# 3. GENERAL HELPERS
# ============================================================

def download_text(url: str) -> str:
    """
    Download text/HTML/CSV from a URL.
    """

    headers = {
        "User-Agent": (
            "Mozilla/5.0 SmartGridForecaster/1.0 "
            "(University research project)"
        )
    }

    response = requests.get(
        url,
        headers=headers,
        timeout=60,
    )

    response.raise_for_status()

    return response.text


def slugify(text: str) -> str:
    """
    Convert text into a simple identifier.
    """

    text = str(text).lower().strip()

    text = re.sub(
        r"[^a-z0-9]+",
        "_",
        text,
    )

    return text.strip("_")


def standardize_events(df: pd.DataFrame) -> pd.DataFrame:
    """
    Make every event dataset use exactly the same columns.
    """

    df = df.copy()

    for column in EVENT_COLUMNS:
        if column not in df.columns:
            df[column] = ""

    # If end_date is missing, use start_date.
    missing_end = (
        df["end_date"].isna()
        | (df["end_date"].astype(str).str.strip() == "")
    )

    df.loc[
        missing_end,
        "end_date"
    ] = df.loc[
        missing_end,
        "start_date"
    ]

    # Standardize dates.
    for column in ["start_date", "end_date"]:

        parsed = pd.to_datetime(
            df[column],
            errors="coerce",
        )

        df[column] = parsed.dt.strftime(
            "%Y-%m-%d"
        )

    return df[EVENT_COLUMNS]


def generate_event_ids(
    df: pd.DataFrame
) -> pd.DataFrame:
    """
    Generate stable event IDs.
    """

    df = df.copy()

    ids = []

    for _, row in df.iterrows():

        base = (
            f"{row['event_type']}_"
            f"{row['start_date']}_"
            f"{row['event_name']}"
        )

        ids.append(
            slugify(base)
        )

    df["event_id"] = ids

    # In case two events accidentally have the same ID.
    duplicate_number = (
        df.groupby("event_id")
        .cumcount()
    )

    duplicate_mask = duplicate_number > 0

    df.loc[
        duplicate_mask,
        "event_id"
    ] = (
        df.loc[
            duplicate_mask,
            "event_id"
        ]
        + "_"
        + duplicate_number[
            duplicate_mask
        ].astype(str)
    )

    return df


def save_dataset(
    df: pd.DataFrame,
    path: Path,
) -> None:

    df.to_csv(
        path,
        index=False,
    )

    print(
        f"Saved {len(df):,} rows -> {path}"
    )


# ============================================================
# 4. GENERAL ELECTION DATA
# ============================================================

def election_fallback() -> pd.DataFrame:
    """
    Fallback values from the official
    House of Commons Library historical election table.

    Used only if HTML extraction fails.
    """

    records = [
        {
            "event_name": "UK General Election 2010",
            "event_type": "general_election",
            "start_date": "2010-05-06",
        },
        {
            "event_name": "UK General Election 2015",
            "event_type": "general_election",
            "start_date": "2015-05-07",
        },
        {
            "event_name": "UK General Election 2017",
            "event_type": "general_election",
            "start_date": "2017-06-08",
        },
        {
            "event_name": "UK General Election 2019",
            "event_type": "general_election",
            "start_date": "2019-12-12",
        },
        {
            "event_name": "UK General Election 2024",
            "event_type": "general_election",
            "start_date": "2024-07-04",
        },
    ]

    df = pd.DataFrame(records)

    df["end_date"] = df["start_date"]

    df["start_time"] = ""
    df["end_time"] = ""

    df["all_day"] = 1

    df["scope"] = "United Kingdom"

    df["importance"] = "high"

    df["availability"] = "scheduled"

    df["use_for_forecast"] = 1

    df["source_type"] = "official"

    df["source_name"] = (
        "House of Commons Library"
    )

    df["source_url"] = (
        PARLIAMENT_ELECTION_URL
    )

    df["notes"] = (
        "Official UK General Election date."
    )

    df = standardize_events(df)

    return generate_event_ids(df)


def extract_general_elections() -> pd.DataFrame:
    """
    Extract UK General Election dates from
    House of Commons Library.
    """

    print(
        "\nExtracting General Elections..."
    )

    try:

        html = download_text(
            PARLIAMENT_ELECTION_URL
        )

        tables = pd.read_html(
            StringIO(html)
        )

        election_table = None

        # Find table containing historical elections.
        for table in tables:

            text = " ".join(
                table
                .astype(str)
                .fillna("")
                .values
                .ravel()
            )

            required_years = [
                "2010",
                "2015",
                "2017",
                "2019",
                "2024",
            ]

            if all(
                year in text
                for year in required_years
            ):
                election_table = table
                break

        if election_table is None:

            raise ValueError(
                "Election table not found."
            )

        records = []

        for _, row in election_table.iterrows():

            values = [
                str(value).strip()
                for value in row.tolist()
            ]

            if len(values) < 2:
                continue

            year_match = re.search(
                r"\b(18|19|20)\d{2}\b",
                values[0],
            )

            if not year_match:
                continue

            year = int(
                year_match.group()
            )

            if year < START_DATE.year:
                continue

            if year > END_DATE.year:
                continue

            date_text = values[1]

            # Remove weekday.
            date_text = re.sub(
                r"^(Monday|Tuesday|Wednesday|"
                r"Thursday|Friday|Saturday|Sunday)\s+",
                "",
                date_text,
                flags=re.IGNORECASE,
            )

            parsed_date = pd.to_datetime(
                f"{date_text} {year}",
                dayfirst=True,
                errors="coerce",
            )

            if pd.isna(parsed_date):
                continue

            records.append(
                {
                    "event_name":
                        f"UK General Election {year}",

                    "event_type":
                        "general_election",

                    "start_date":
                        parsed_date,

                    "end_date":
                        parsed_date,

                    "start_time":
                        "",

                    "end_time":
                        "",

                    "all_day":
                        1,

                    "scope":
                        "United Kingdom",

                    "importance":
                        "high",

                    "availability":
                        "scheduled",

                    "use_for_forecast":
                        1,

                    "source_type":
                        "official",

                    "source_name":
                        "House of Commons Library",

                    "source_url":
                        PARLIAMENT_ELECTION_URL,

                    "notes":
                        "Extracted from official historical election table.",
                }
            )

        df = pd.DataFrame(records)

        if df.empty:
            raise ValueError(
                "No election records extracted."
            )

        df = standardize_events(df)

        return generate_event_ids(df)

    except Exception as error:

        print(
            "Election extraction failed:"
        )

        print(error)

        print(
            "Using verified fallback dates."
        )

        return election_fallback()


# ============================================================
# 5. FOOTBALL DATA
# ============================================================

def extract_major_football_events() -> pd.DataFrame:
    """
    Download historical international football results.

    Keep only England:
        - FIFA World Cup finals tournament
        - UEFA Euro finals tournament

    Qualification matches and friendlies are excluded.
    """

    print(
        "\nExtracting England football events..."
    )

    csv_text = download_text(
        FOOTBALL_RESULTS_URL
    )

    df = pd.read_csv(
        StringIO(csv_text)
    )

    df["date"] = pd.to_datetime(
        df["date"],
        errors="coerce",
    )

    # England must be playing.
    england_mask = (
        (df["home_team"] == "England")
        |
        (df["away_team"] == "England")
    )

    # Only major final tournaments.
    tournament_mask = (
        df["tournament"]
        .isin(
            [
                "FIFA World Cup",
                "UEFA Euro",
            ]
        )
    )

    date_mask = (
        (df["date"] >= START_DATE)
        &
        (df["date"] <= END_DATE)
    )

    df = df[
        england_mask
        &
        tournament_mask
        &
        date_mask
    ].copy()

    events = []

    for _, row in df.iterrows():

        if row["home_team"] == "England":
            opponent = row["away_team"]
        else:
            opponent = row["home_team"]

        event_name = (
            f"England vs {opponent} "
            f"- {row['tournament']}"
        )

        events.append(
            {
                "event_name":
                    event_name,

                "event_type":
                    "major_football",

                "start_date":
                    row["date"],

                "end_date":
                    row["date"],

                # This source contains the match date
                # but not reliable kickoff-time information.
                "start_time":
                    "",

                "end_time":
                    "",

                "all_day":
                    1,

                "scope":
                    "Great Britain",

                "importance":
                    "high",

                "availability":
                    "scheduled",

                "use_for_forecast":
                    1,

                "source_type":
                    "secondary_structured",

                "source_name":
                    "martj42 international_results",

                "source_url":
                    FOOTBALL_RESULTS_URL,

                "notes":
                    (
                        "Major England tournament match. "
                        "Verify final selected matches against "
                        "England Football / UEFA / FIFA."
                    ),
            }
        )

    events_df = pd.DataFrame(events)

    events_df = standardize_events(
        events_df
    )

    return generate_event_ids(
        events_df
    )


# ============================================================
# 6. COVID-19 EVENT DATA
# ============================================================

def create_covid_events() -> pd.DataFrame:
    """
    Important COVID restriction periods.

    IMPORTANT:
    These mainly represent England restrictions.

    Because NESO represents Great Britain,
    keep 'scope' in the dataset rather than
    pretending all devolved nations had identical rules.
    """

    print(
        "\nCreating COVID event data..."
    )

    records = [
        {
            "event_name":
                "First COVID-19 national stay-at-home phase",

            "event_type":
                "covid_lockdown",

            "start_date":
                "2020-03-23",

            "end_date":
                "2020-05-12",

            "scope":
                "England",

            "importance":
                "very_high",

            "source_name":
                "Office for National Statistics",

            "source_url":
                (
                    "https://www.ons.gov.uk/economy/"
                    "economicoutputandproductivity/output/"
                    "articles/coronavirushowpeopleandbusinesses"
                    "haveadaptedtolockdowns/2021-03-19"
                ),

            "notes":
                (
                    "Modelled initial stay-at-home phase. "
                    "Review boundaries if modelling "
                    "England, Scotland and Wales separately."
                ),
        },

        {
            "event_name":
                "Second COVID-19 national lockdown",

            "event_type":
                "covid_lockdown",

            "start_date":
                "2020-11-05",

            "end_date":
                "2020-12-01",

            "scope":
                "England",

            "importance":
                "very_high",

            "source_name":
                "GOV.UK",

            "source_url":
                (
                    "https://gds.blog.gov.uk/"
                    "2022/07/25/"
                    "2-years-of-covid-19-on-gov-uk/"
                ),

            "notes":
                "Second national lockdown in England.",
        },

        {
            "event_name":
                "Third COVID-19 national lockdown",

            "event_type":
                "covid_lockdown",

            "start_date":
                "2021-01-05",

            "end_date":
                "2021-03-07",

            "scope":
                "England",

            "importance":
                "very_high",

            "source_name":
                "GOV.UK",

            "source_url":
                (
                    "https://www.gov.uk/government/"
                    "publications/covid-19-response-spring-2021/"
                    "covid-19-response-spring-2021-summary"
                ),

            "notes":
                (
                    "Initial third-lockdown phase "
                    "before Step 1 reopening."
                ),
        },

        {
            "event_name":
                "COVID Roadmap Step 1",

            "event_type":
                "covid_reopening",

            "start_date":
                "2021-03-08",

            "end_date":
                "2021-03-08",

            "scope":
                "England",

            "importance":
                "high",

            "source_name":
                "GOV.UK",

            "source_url":
                (
                    "https://www.gov.uk/government/"
                    "publications/covid-19-response-spring-2021/"
                    "covid-19-response-spring-2021-summary"
                ),

            "notes":
                "Schools reopened under Step 1.",
        },

        {
            "event_name":
                "COVID Roadmap Step 2",

            "event_type":
                "covid_reopening",

            "start_date":
                "2021-04-12",

            "end_date":
                "2021-04-12",

            "scope":
                "England",

            "importance":
                "high",

            "source_name":
                "GOV.UK",

            "source_url":
                (
                    "https://www.gov.uk/government/"
                    "publications/covid-19-response-spring-2021/"
                    "covid-19-response-spring-2021-summary"
                ),

            "notes":
                "Step 2 reopening.",
        },

        {
            "event_name":
                "COVID Roadmap Step 3",

            "event_type":
                "covid_reopening",

            "start_date":
                "2021-05-17",

            "end_date":
                "2021-05-17",

            "scope":
                "England",

            "importance":
                "high",

            "source_name":
                "GOV.UK",

            "source_url":
                (
                    "https://www.gov.uk/government/"
                    "publications/covid-19-response-spring-2021/"
                    "covid-19-response-spring-2021-summary"
                ),

            "notes":
                "Step 3 reopening.",
        },
    ]

    df = pd.DataFrame(
        records
    )

    df["start_time"] = ""
    df["end_time"] = ""

    df["all_day"] = 1

    df["availability"] = (
        "policy_announced"
    )

    df["use_for_forecast"] = 1

    df["source_type"] = "official"

    df = standardize_events(df)

    return generate_event_ids(df)


# ============================================================
# 7. ROYAL / NATIONAL EVENTS
# ============================================================

def create_royal_events() -> pd.DataFrame:
    """
    Major Royal / national events from 2010 onward.
    """

    print(
        "\nCreating Royal event data..."
    )

    records = [
        {
            "event_name":
                "Royal Wedding - William and Catherine",

            "event_type":
                "royal_event",

            "start_date":
                "2011-04-29",

            "end_date":
                "2011-04-29",

            "start_time":
                "11:00",

            "end_time":
                "",

            "all_day":
                0,

            "scope":
                "United Kingdom",

            "importance":
                "high",

            "availability":
                "scheduled",

            "use_for_forecast":
                1,

            "source_type":
                "official",

            "source_name":
                "The Royal Family",

            "source_url":
                (
                    "https://www.royal.uk/"
                    "wedding-prince-william-and-"
                    "miss-catherine-middleton"
                ),

            "notes":
                "Wedding service began at 11:00.",
        },

        {
            "event_name":
                "Queen Elizabeth II Diamond Jubilee Weekend",

            "event_type":
                "royal_event",

            "start_date":
                "2012-06-02",

            "end_date":
                "2012-06-05",

            "start_time":
                "",

            "end_time":
                "",

            "all_day":
                1,

            "scope":
                "United Kingdom",

            "importance":
                "high",

            "availability":
                "scheduled",

            "use_for_forecast":
                1,

            "source_type":
                "official",

            "source_name":
                "The Royal Family",

            "source_url":
                (
                    "https://www.royal.uk/"
                    "queen%E2%80%99s-diamond-jubilee-2012"
                ),

            "notes":
                "Official Jubilee weekend 2-5 June.",
        },

        {
            "event_name":
                "Queen Elizabeth II Platinum Jubilee Weekend",

            "event_type":
                "royal_event",

            "start_date":
                "2022-06-02",

            "end_date":
                "2022-06-05",

            "start_time":
                "",

            "end_time":
                "",

            "all_day":
                1,

            "scope":
                "United Kingdom",

            "importance":
                "high",

            "availability":
                "scheduled",

            "use_for_forecast":
                1,

            "source_type":
                "official",

            "source_name":
                "The Royal Family",

            "source_url":
                (
                    "https://www.royal.uk/"
                    "platinum-jubilee-weekend"
                ),

            "notes":
                "Four-day Platinum Jubilee weekend.",
        },

        {
            "event_name":
                "Death of Queen Elizabeth II",

            "event_type":
                "national_mourning",

            "start_date":
                "2022-09-08",

            "end_date":
                "2022-09-08",

            "start_time":
                "",

            "end_time":
                "",

            "all_day":
                1,

            "scope":
                "United Kingdom",

            "importance":
                "very_high",

            "availability":
                "unplanned",

            # This event was not known in advance.
            # Do not use the occurrence itself as
            # a normal future-known predictor.
            "use_for_forecast":
                0,

            "source_type":
                "official",

            "source_name":
                "The Royal Family",

            "source_url":
                (
                    "https://www.royal.uk/"
                    "queen-elizabeth"
                ),

            "notes":
                (
                    "Keep for anomaly analysis. "
                    "The death itself was not a "
                    "future-known event."
                ),
        },

        {
            "event_name":
                "State Funeral of Queen Elizabeth II",

            "event_type":
                "royal_event",

            "start_date":
                "2022-09-19",

            "end_date":
                "2022-09-19",

            "start_time":
                "11:00",

            "end_time":
                "",

            "all_day":
                0,

            "scope":
                "United Kingdom",

            "importance":
                "very_high",

            "availability":
                "scheduled",

            "use_for_forecast":
                1,

            "source_type":
                "official",

            "source_name":
                "The Royal Family",

            "source_url":
                (
                    "https://www.royal.uk/"
                    "news-and-activity/2022-09-19/"
                    "the-state-funeral-for-her-"
                    "majesty-the-queen"
                ),

            "notes":
                "State Funeral began at 11:00.",
        },

        {
            "event_name":
                "Coronation of King Charles III",

            "event_type":
                "royal_event",

            "start_date":
                "2023-05-06",

            "end_date":
                "2023-05-06",

            "start_time":
                "11:00",

            "end_time":
                "",

            "all_day":
                0,

            "scope":
                "United Kingdom",

            "importance":
                "very_high",

            "availability":
                "scheduled",

            "use_for_forecast":
                1,

            "source_type":
                "official",

            "source_name":
                "The Royal Family",

            "source_url":
                (
                    "https://www.royal.uk/"
                    "coronation-weekend"
                ),

            "notes":
                "Coronation service started at 11:00.",
        },
    ]

    df = pd.DataFrame(
        records
    )

    df = standardize_events(df)

    return generate_event_ids(df)


# ============================================================
# 8. GRID / UNPLANNED ANOMALY EVENTS
# ============================================================

def create_anomaly_events() -> pd.DataFrame:
    """
    Unplanned system events.

    IMPORTANT:
    These should normally NOT be features used
    to predict future demand because they were
    not known before they occurred.

    They are useful for:
        - EDA
        - anomaly identification
        - excluding unusual periods
        - model-error analysis
    """

    print(
        "\nCreating anomaly event data..."
    )

    records = [
        {
            "event_name":
                "Great Britain Power System Disruption",

            "event_type":
                "grid_disruption",

            "start_date":
                "2019-08-09",

            "end_date":
                "2019-08-09",

            "start_time":
                "16:52",

            "end_time":
                "",

            "all_day":
                0,

            "scope":
                "Great Britain",

            "importance":
                "very_high",

            "availability":
                "unplanned",

            # CRITICAL:
            "use_for_forecast":
                0,

            "source_type":
                "official",

            "source_name":
                "Ofgem",

            "source_url":
                (
                    "https://www.ofgem.gov.uk/"
                    "publications/investigation-"
                    "9-august-2019-power-outage"
                ),

            "notes":
                (
                    "Unplanned grid event. "
                    "Use as anomaly label, "
                    "not a future-known forecasting feature."
                ),
        }
    ]

    df = pd.DataFrame(
        records
    )

    df = standardize_events(df)

    return generate_event_ids(df)


# ============================================================
# 9. COMBINE ALL EVENT DATA
# ============================================================

def combine_event_datasets(
    event_datasets: list[pd.DataFrame],
) -> pd.DataFrame:

    combined = pd.concat(
        event_datasets,
        ignore_index=True,
    )

    combined = combined.drop_duplicates(
        subset=[
            "event_name",
            "event_type",
            "start_date",
            "end_date",
        ]
    )

    combined["start_date_dt"] = (
        pd.to_datetime(
            combined["start_date"],
            errors="coerce",
        )
    )

    combined = combined[
        (
            combined["start_date_dt"]
            >= START_DATE
        )
        &
        (
            combined["start_date_dt"]
            <= END_DATE
        )
    ]

    combined = combined.sort_values(
        [
            "start_date_dt",
            "event_type",
        ]
    )

    combined = combined.drop(
        columns=[
            "start_date_dt"
        ]
    )

    combined = combined.reset_index(
        drop=True
    )

    return combined


# ============================================================
# 10. EXPAND EVENTS INTO DAILY ROWS
# ============================================================

def expand_events_to_daily(
    events: pd.DataFrame,
    forecast_only: bool = True,
) -> pd.DataFrame:
    """
    Convert:

        2022-06-02 -> 2022-06-05

    into:

        2022-06-02
        2022-06-03
        2022-06-04
        2022-06-05
    """

    working = events.copy()

    if forecast_only:

        working = working[
            working["use_for_forecast"] == 1
        ]

    rows = []

    for _, event in working.iterrows():

        start = pd.to_datetime(
            event["start_date"]
        )

        end = pd.to_datetime(
            event["end_date"]
        )

        dates = pd.date_range(
            start=start,
            end=end,
            freq="D",
        )

        for date in dates:

            rows.append(
                {
                    "date":
                        date,

                    "event_id":
                        event["event_id"],

                    "event_name":
                        event["event_name"],

                    "event_type":
                        event["event_type"],

                    "importance":
                        event["importance"],

                    "scope":
                        event["scope"],
                }
            )

    return pd.DataFrame(rows)


# ============================================================
# 11. CREATE ML DAILY EVENT FEATURES
# ============================================================

def create_daily_event_features(
    events: pd.DataFrame,
) -> pd.DataFrame:

    daily_long = expand_events_to_daily(
        events,
        forecast_only=True,
    )

    if daily_long.empty:

        return pd.DataFrame(
            columns=["date"]
        )

    importance_values = {
        "low": 1,
        "medium": 2,
        "high": 3,
        "very_high": 4,
    }

    daily_long[
        "importance_score"
    ] = (
        daily_long["importance"]
        .map(importance_values)
        .fillna(0)
        .astype(int)
    )

    # --------------------------------------------------------
    # Number of events per day
    # --------------------------------------------------------

    summary = (
        daily_long
        .groupby("date")
        .agg(
            event_count=(
                "event_id",
                "nunique",
            ),

            max_event_importance=(
                "importance_score",
                "max",
            ),

            event_names=(
                "event_name",
                lambda values:
                    " | ".join(
                        sorted(
                            set(values)
                        )
                    ),
            ),
        )
        .reset_index()
    )

    # --------------------------------------------------------
    # Binary feature for each event type
    # --------------------------------------------------------

    event_flags = pd.crosstab(
        daily_long["date"],
        daily_long["event_type"],
    )

    event_flags = (
        event_flags > 0
    ).astype(int)

    event_flags.columns = [
        f"is_{slugify(column)}"
        for column
        in event_flags.columns
    ]

    event_flags = (
        event_flags
        .reset_index()
    )

    output = summary.merge(
        event_flags,
        on="date",
        how="left",
    )

    # General event indicator.
    output["is_event_day"] = (
        output["event_count"] > 0
    ).astype(int)

    output = output.sort_values(
        "date"
    )

    return output


# ============================================================
# 12. VALIDATION
# ============================================================

def validate_events(
    events: pd.DataFrame
) -> None:

    print(
        "\n=============================="
    )

    print(
        "EVENT DATA VALIDATION"
    )

    print(
        "=============================="
    )

    print(
        f"Total events: {len(events):,}"
    )

    print()

    print(
        "Events by type:"
    )

    print(
        events[
            "event_type"
        ].value_counts()
    )

    print()

    print(
        "Events by source type:"
    )

    print(
        events[
            "source_type"
        ].value_counts()
    )

    print()

    print(
        "Forecast-safe:"
    )

    print(
        events[
            "use_for_forecast"
        ].value_counts()
    )

    # --------------------------------------------------------
    # Missing dates
    # --------------------------------------------------------

    missing_dates = events[
        events["start_date"].isna()
    ]

    if not missing_dates.empty:

        print(
            "\nWARNING: Missing dates found."
        )

        print(
            missing_dates[
                [
                    "event_name",
                    "event_type",
                ]
            ]
        )

    # --------------------------------------------------------
    # Invalid date ranges
    # --------------------------------------------------------

    start = pd.to_datetime(
        events["start_date"],
        errors="coerce",
    )

    end = pd.to_datetime(
        events["end_date"],
        errors="coerce",
    )

    invalid_range = events[
        end < start
    ]

    if not invalid_range.empty:

        print(
            "\nWARNING: Invalid date ranges."
        )

        print(
            invalid_range
        )

    print(
        "\nValidation complete."
    )


# ============================================================
# 13. MAIN PIPELINE
# ============================================================

def main():

    print(
        "====================================="
    )

    print(
        "SMART GRID EVENT DATA PIPELINE"
    )

    print(
        "====================================="
    )

    print(
        f"Start date : {START_DATE.date()}"
    )

    print(
        f"End date   : {END_DATE.date()}"
    )

    # --------------------------------------------------------
    # Extract / create source datasets
    # --------------------------------------------------------

    elections = (
        extract_general_elections()
    )

    football = (
        extract_major_football_events()
    )

    covid = (
        create_covid_events()
    )

    royal = (
        create_royal_events()
    )

    anomalies = (
        create_anomaly_events()
    )

    # --------------------------------------------------------
    # Save raw source files
    # --------------------------------------------------------

    save_dataset(
        elections,
        RAW_DIR / "elections.csv",
    )

    save_dataset(
        football,
        RAW_DIR / "football_events.csv",
    )

    save_dataset(
        covid,
        RAW_DIR / "covid_events.csv",
    )

    save_dataset(
        royal,
        RAW_DIR / "royal_events.csv",
    )

    save_dataset(
        anomalies,
        RAW_DIR / "anomaly_events.csv",
    )

    # --------------------------------------------------------
    # Combine
    # --------------------------------------------------------

    events = combine_event_datasets(
        [
            elections,
            football,
            covid,
            royal,
            anomalies,
        ]
    )

    save_dataset(
        events,
        PROCESSED_DIR / "gb_events.csv",
    )

    # --------------------------------------------------------
    # Generate ML-safe daily features
    # --------------------------------------------------------

    daily_features = (
        create_daily_event_features(
            events
        )
    )

    save_dataset(
        daily_features,
        PROCESSED_DIR
        / "gb_event_daily_features.csv",
    )

    # --------------------------------------------------------
    # Validation
    # --------------------------------------------------------

    validate_events(
        events
    )

    print(
        "\nPipeline completed successfully."
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    main()